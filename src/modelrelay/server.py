"""`modelrelay serve`: a local OpenAI-compatible endpoint backed by a Relay.

Lets programs in any language (Node, Go, a browser app...) use modelrelay without
knowing about gateways, tokens or adapters: they talk to plain /v1/chat/completions
on localhost, and the config decides where calls really go.

    from modelrelay.server import make_server
    server = make_server(port=0)          # port 0 = any free port
    print(server.url)                     # http://127.0.0.1:54321/v1
    server.serve_forever()

Only the standard library is used. By default it listens on 127.0.0.1 without auth;
pass `api_key` (or set $MODELRELAY_SERVE_KEY) to require `Authorization: Bearer <key>`.

Apps identify themselves with the `X-Modelrelay-App: <app>` header; model names are then
resolved with [apps.<app>.models] first, so one server can give each app its own models.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .errors import ModelRelayError, ProviderError
from .relay import Relay
from .types import Image, Response

log = logging.getLogger("modelrelay.server")

APP_HEADER = "X-Modelrelay-App"

# Request fields handled here; everything else goes to the provider as a param.
_OWN_FIELDS = {"model", "messages", "tools", "stream", "stream_options"}


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, relay: Relay, api_key: str | None):
        super().__init__(address, _Handler)
        self.relay = relay
        self.api_key = api_key

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"


def make_server(relay: Relay | None = None, *, host: str = "127.0.0.1", port: int = 8765,
                api_key: str | None = None, profile: str | None = None) -> RelayServer:
    relay = relay or Relay(profile=profile)
    api_key = api_key or os.environ.get("MODELRELAY_SERVE_KEY") or None
    return RelayServer((host, port), relay, api_key)


def serve(**kwargs) -> None:
    server = make_server(**kwargs)
    print(f"modelrelay serving {server.url} (config: {server.relay.config.source or 'built-in defaults'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class _Handler(BaseHTTPRequestHandler):
    server: RelayServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # route through logging instead of stderr
        log.info("%s %s", self.address_string(), fmt % args)

    # ---- routes --------------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health"):
            return self._json(200, {"ok": True, "config": str(self.server.relay.config.source or "")})
        if not self._authorized():
            return
        if path == "/v1/models":
            names = list(self.server.relay.config.models_for(self._app()))
            return self._json(200, {"object": "list", "data": [
                {"id": n, "object": "model", "owned_by": "modelrelay"} for n in names]})
        self._error(404, f"Not found: {self.path}")

    def do_POST(self):
        if not self._authorized():
            return
        path = self.path.split("?")[0].rstrip("/")
        if path != "/v1/chat/completions":
            return self._error(404, f"Not found: {self.path}")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:  # invalid JSON or not UTF-8
            return self._error(400, f"Invalid JSON body (must be UTF-8 JSON): {e}")
        if not body.get("model") or not body.get("messages"):
            return self._error(400, "`model` and `messages` are required")

        model, messages, tools = body["model"], body["messages"], body.get("tools") or None
        params = {k: v for k, v in body.items() if k not in _OWN_FIELDS}
        app = self._app()
        try:
            if body.get("stream"):
                return self._stream(model, messages, tools, params, app)
            resp = self.server.relay.chat(messages, model=model, tools=tools, app=app, **params)
        except ModelRelayError as e:
            return self._relay_error(e)
        except (ValueError, TypeError) as e:  # malformed messages/tools
            return self._error(400, str(e))
        self._json(200, _completion(resp, model))

    # ---- streaming (server-sent events) ----------------------------------------------

    def _stream(self, model, messages, tools, params, app=None):
        cid, created = _completion_id(), int(time.time())
        started = sent_text = False
        try:
            for ev in self.server.relay.stream(messages, model=model, tools=tools, app=app, **params):
                if not started:
                    self._start_sse()
                    started = True
                if ev.type == "delta" and ev.text:
                    sent_text = True
                    self._sse(_chunk(cid, created, model, {"content": ev.text}))
                elif ev.type == "done":
                    resp = ev.response or Response()
                    delta = {} if sent_text or not resp.text else {"content": resp.text}
                    if resp.tool_calls:
                        delta["tool_calls"] = [dict(tc.to_openai(), index=i) for i, tc in enumerate(resp.tool_calls)]
                    if resp.images:
                        delta["images"] = [_image_part(img) for img in resp.images]
                    if delta:
                        self._sse(_chunk(cid, created, model, delta))
                    final = _chunk(cid, created, model, {}, resp.finish_reason or "stop")
                    if resp.usage:
                        final["usage"] = _usage(resp)
                    self._sse(final)
        except ModelRelayError as e:
            if not started:
                return self._relay_error(e)
            self._sse({"error": {"message": str(e), "type": type(e).__name__}})
        except (ValueError, TypeError) as e:
            if not started:
                return self._error(400, str(e))
            self._sse({"error": {"message": str(e), "type": type(e).__name__}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, data: dict):
        self.wfile.write(b"data: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n")
        self.wfile.flush()

    # ---- helpers -----------------------------------------------------------------------

    def _app(self) -> str | None:
        return (self.headers.get(APP_HEADER) or "").strip() or None

    def _authorized(self) -> bool:
        key = self.server.api_key
        if not key or self.headers.get("Authorization", "") == f"Bearer {key}":
            return True
        self._error(401, "Missing or wrong bearer token")
        return False

    def _relay_error(self, e: ModelRelayError):
        status = getattr(e, "status", None) if isinstance(e, ProviderError) else None
        code = status if status and 400 <= status < 500 else 502
        log.warning("relay error: %s", e)
        self._json(code, {"error": {"message": str(e), "type": type(e).__name__, "upstream_status": status}})

    def _error(self, status: int, message: str):
        self._json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def _json(self, status: int, data: dict):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _image_part(img: Image) -> dict:
    url = img.url or f"data:{img.mime_type};base64,{base64.b64encode(img.data).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


def _usage(resp: Response) -> dict:
    i, o = resp.usage.input_tokens or 0, resp.usage.output_tokens or 0
    return {"prompt_tokens": i, "completion_tokens": o, "total_tokens": i + o}


def _completion(resp: Response, model: str) -> dict:
    message: dict = {"role": "assistant", "content": resp.text or None}
    if resp.tool_calls:
        message["tool_calls"] = [tc.to_openai() for tc in resp.tool_calls]
    if resp.images:
        message["images"] = [_image_part(img) for img in resp.images]
    out = {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.model or model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": resp.finish_reason or ("tool_calls" if resp.tool_calls else "stop")}],
    }
    if resp.usage:
        out["usage"] = _usage(resp)
    if resp.trace_id:
        out["trace_id"] = resp.trace_id
    return out


def _chunk(cid: str, created: int, model: str, delta: dict, finish: str | None = None) -> dict:
    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
