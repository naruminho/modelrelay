"""A local fake gateway for developing and testing without spending tokens.

It imitates a typical corporate setup:
  POST /auth/token          client id + secret -> {"data": {"token": ...}}, no expiry info;
                            the token silently dies after `token_ttl` seconds (HTTP 403).
  POST /v1/chat/completions OpenAI-compatible, with stream; requests slower than
                            `request_limit` seconds fail with 504 (model "mock-slow").
  POST /v1/jobs             queues a request -> {"data": {"execution": {"id": ...}}}
  GET  /v1/jobs/<id>        STARTED -> IN_PROGRESS -> FINISHED | FAILED, nested payload.

Fake models: anything echoes the last user message. Special models:
  mock-image    returns an image
  mock-slow     exceeds the time limit (504)
  mock-cut      stream is cut in the middle (no finish, no [DONE])
  mock-badtool  calls a tool with arguments that are not JSON
  mock-fail     job ends FAILED
  mock-flaky    first poll of the job answers HTTP 500
  mock-weird    job reports an undocumented state ("PAUSED")
If tools are sent and the user message mentions "tool", the first tool is called.

Run:  python -m modelrelay.testing.mock_server --port 8000
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

# 1x1 transparent PNG
PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class MockState:
    def __init__(self, token_ttl=120.0, request_limit=30.0, polls_to_finish=3, static_keys=("test-key",)):
        self.token_ttl = token_ttl
        self.request_limit = request_limit
        self.polls_to_finish = polls_to_finish
        self.static_keys = set(static_keys)
        self.tokens: dict[str, float] = {}
        self.jobs: dict[str, dict] = {}
        self.token_requests = 0
        self.last_headers: dict = {}
        self.lock = threading.Lock()


def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            return " ".join(p.get("text", "") for p in c or [] if p.get("type") == "text")
    return ""


def fake_completion(body: dict) -> dict:
    messages = body.get("messages") or []
    model = body.get("model", "mock")
    tools = body.get("tools") or []
    last = messages[-1] if messages else {}
    msg: dict = {"role": "assistant", "content": None}
    finish = "stop"
    user_text = _last_user_text(messages)
    attachments = sum(
        1 for m in messages if isinstance(m.get("content"), list)
        for p in m["content"] if p.get("type") in ("image_url", "file")
    )

    if tools and last.get("role") == "user" and "tool" in user_text.lower():
        fn = tools[0]["function"]
        args = "{not json" if model == "mock-badtool" else json.dumps({"a": 2, "b": 3})
        msg["tool_calls"] = [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": fn["name"], "arguments": args},
        }]
        finish = "tool_calls"
    elif last.get("role") == "tool":
        msg["content"] = f"tool said: {last.get('content')}"
    elif model == "mock-image":
        msg["content"] = "here is your image"
        msg["images"] = [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}]
    else:
        msg["content"] = f"echo: {user_text}" + (f" [{attachments} attachment(s)]" if attachments else "")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        # ---- helpers ----
        def _body(self) -> dict:
            raw = self._raw
            if "application/x-www-form-urlencoded" in (self.headers.get("Content-Type") or ""):
                return {k: v[0] for k, v in parse_qs(raw.decode()).items()}
            return json.loads(raw or b"{}")

        def _send(self, status: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            state.last_headers = dict(self.headers)
            header = self.headers.get("Authorization", "")
            token = header[7:] if header.startswith("Bearer ") else ""
            with state.lock:
                created = state.tokens.get(token)
            if token in state.static_keys or (created is not None and time.monotonic() - created < state.token_ttl):
                return True
            self._send(403, {"detail": "permission denied"})
            return False

        # ---- routes ----
        def do_POST(self):
            # Always consume the body first, so keep-alive connections stay in sync.
            length = int(self.headers.get("Content-Length") or 0)
            self._raw = self.rfile.read(length) if length else b""
            if self.path == "/auth/token":
                return self._token()
            if not self._authorized():
                return
            if self.path == "/v1/chat/completions":
                return self._chat()
            if self.path == "/v1/jobs":
                return self._submit()
            self._send(404, {"detail": "not found"})

        def do_GET(self):
            if not self._authorized():
                return
            if self.path.startswith("/v1/jobs/"):
                return self._poll(self.path.rsplit("/", 1)[1])
            if self.path == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": "mock-fast"}, {"id": "mock-image"}]})
            self._send(404, {"detail": "not found"})

        def _token(self):
            body = self._body()
            if not body.get("client_id") or not body.get("client_secret"):
                return self._send(400, {"detail": "client_id and client_secret are required"})
            token = f"mock-{uuid.uuid4().hex}"
            with state.lock:
                state.tokens[token] = time.monotonic()
                state.token_requests += 1
            self._send(200, {"data": {"token": token}})

        def _chat(self):
            body = self._body()
            if body.get("model") == "mock-slow":
                time.sleep(state.request_limit + 0.05)
                return self._send(504, {"error": {"message": "upstream request timeout"}})
            completion = fake_completion(body)
            if not body.get("stream"):
                return self._send(200, completion)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            msg = completion["choices"][0]["message"]

            def emit(delta, finish=None, usage=None):
                chunk = {"model": completion["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage:
                    chunk["usage"] = usage
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            self.wfile.write(b": keep-alive\n\n")
            if body.get("model") == "mock-cut":
                emit({"content": "this answer gets cut "})
                self.close_connection = True
                return
            for word in (msg.get("content") or "").split(" "):
                emit({"content": word + " "})
            for i, tc in enumerate(msg.get("tool_calls") or []):
                args = tc["function"]["arguments"]
                half = len(args) // 2
                emit({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                      "function": {"name": tc["function"]["name"], "arguments": args[:half]}}]})
                emit({"tool_calls": [{"index": i, "function": {"arguments": args[half:]}}]})
            if msg.get("images"):
                emit({"images": msg["images"]})
            emit({}, finish=completion["choices"][0]["finish_reason"], usage=completion["usage"])
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

        def _submit(self):
            body = self._body()
            job_id = uuid.uuid4().hex[:12]
            with state.lock:
                state.jobs[job_id] = {"polls": 0, "inputs": body.get("inputs") or {}}
            self._send(200, {"meta": {"code": "ACCEPTED"}, "data": {"execution": {"id": job_id}}})

        def _poll(self, job_id):
            with state.lock:
                job = state.jobs.get(job_id)
                if job:
                    job["polls"] += 1
            if not job:
                return self._send(404, {"detail": "unknown execution"})
            model = job["inputs"].get("model")
            if model == "mock-flaky" and job["polls"] == 1:
                return self._send(500, {"detail": "temporary failure"})
            execution = {"id": job_id}
            payload = {"meta": {"code": "OK"}, "data": {"execution": execution, "output": {}}}
            if model == "mock-weird":
                execution["state"] = "PAUSED"
            elif job["polls"] < state.polls_to_finish:
                execution["state"] = "STARTED" if job["polls"] == 1 else "IN_PROGRESS"
            elif model == "mock-fail":
                execution["state"] = "FAILED"
                payload["data"]["output"] = {"error": {"message": "model exploded"}}
            else:
                execution["state"] = "FINISHED"
                payload["data"]["output"] = {"result": fake_completion(job["inputs"])}
            self._send(200, payload)

    return Handler


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass  # clients closing connections early (e.g. cut streams) are expected here


def start(host="127.0.0.1", port=0, **state_options) -> tuple[ThreadingHTTPServer, MockState, str]:
    """Starts the mock in a background thread. Returns (server, state, root_url)."""
    state = MockState(**state_options)
    server = _QuietServer((host, port), make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server, state, f"http://{host}:{server.server_address[1]}"


def main():
    parser = argparse.ArgumentParser(description="modelrelay mock gateway")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token-ttl", type=float, default=120, help="seconds a token lives")
    parser.add_argument("--request-limit", type=float, default=30, help="seconds before 'mock-slow' fails")
    parser.add_argument("--polls-to-finish", type=int, default=3)
    args = parser.parse_args()
    server, _, url = start(port=args.port, token_ttl=args.token_ttl,
                           request_limit=args.request_limit, polls_to_finish=args.polls_to_finish)
    print(f"Mock gateway on {url}  (static key: test-key)  Ctrl+C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
