from __future__ import annotations

import time
import uuid
import warnings
from typing import Iterator

import httpx

from ._util import load_object, log
from .auth import BearerAuth, build_token_provider
from .config import Config
from .errors import InvalidToolCall, ModelRelayError
from .messages import normalize_messages
from .tools import normalize_tools, parse_emulated, to_emulated
from .transports import BUILTIN_TRANSPORTS, Transport
from .types import ChatRequest, Event, Response


class Relay:
    """The entry point. Your code only talks to this; the config decides where calls go.

        relay = Relay()                      # reads ~/.modelrelay/config.toml
        relay = Relay(profile="openai")      # reads ~/.modelrelay/openai.toml
        relay = Relay(app="wotan")           # model names resolved with [apps.wotan.models] first
        resp = relay.chat("hi", model="gpt-4o")
        for ev in relay.stream(messages, model="gpt-4o"): ...
    """

    def __init__(self, config: Config | None = None, *, config_path=None, profile: str | None = None,
                 app: str | None = None, http: httpx.Client | None = None, **overrides):
        self.config = config or Config.load(config_path, profile, **overrides)
        self.app = app
        log.debug("config: %s", self.config.source or "built-in defaults")
        self.http = http or _make_client(self.config)
        self.auth = BearerAuth(build_token_provider(self.config, self.http))
        self._transports: dict[str, Transport] = {}

    # ---- public API --------------------------------------------------------------

    def chat(self, messages, *, model: str, tools: list[dict] | None = None, files=None, app: str | None = None,
             **params) -> Response:
        """Sends the conversation and waits for the full response.

        messages: a string or a list of {"role", "content"} dicts (you own the history).
        files:    local image/PDF paths attached to the last user message.
        app:      which [apps.<app>.models] to use for this call (default: the Relay's app).
        params:   passed to the provider as-is (temperature, max_tokens, ...).
        """
        transport = self.transport(self.config.transport)
        req, emulated = self._request(transport, messages, model, tools, files, params, app)
        start = time.monotonic()
        log.info("%s chat via %s model=%s messages=%d", req.trace_id, transport.name, req.model, len(req.messages))
        try:
            response = transport.complete(req)
            response = _finish(response, req, emulated)
        except ModelRelayError as e:
            _tag(e, req, transport)
            log.error("%s chat failed after %.1fs: %s", req.trace_id, time.monotonic() - start, e)
            raise
        log.info("%s chat done in %.1fs", req.trace_id, time.monotonic() - start)
        return response

    def stream(self, messages, *, model: str, tools: list[dict] | None = None, files=None, app: str | None = None,
               **params) -> Iterator[Event]:
        """Same as chat(), but yields Events while waiting. The last one is always "done".

        With a text-streaming transport you get "delta" events; with a job transport you get
        "queued" / "running" status events instead. Check `supports_text_stream` if the UI cares.
        """
        transport = self.transport(self.config.stream_transport or self.config.transport)
        req, emulated = self._request(transport, messages, model, tools, files, params, app)
        start = time.monotonic()
        log.info("%s stream via %s model=%s messages=%d", req.trace_id, transport.name, req.model, len(req.messages))
        try:
            for event in transport.stream(req):
                event.trace_id = req.trace_id
                if event.type == "done":
                    event.response = _finish(event.response, req, emulated)
                yield event
        except ModelRelayError as e:
            _tag(e, req, transport)
            log.error("%s stream failed after %.1fs: %s", req.trace_id, time.monotonic() - start, e)
            raise
        log.info("%s stream done in %.1fs", req.trace_id, time.monotonic() - start)

    @property
    def supports_text_stream(self) -> bool:
        return self.transport(self.config.stream_transport or self.config.transport).supports_text_stream

    def transport(self, name: str) -> Transport:
        if name not in self._transports:
            cls = load_object(name, "modelrelay.transports", BUILTIN_TRANSPORTS)
            self._transports[name] = cls(self.http, self.auth, self.config, self.config.transport_options(name))
        return self._transports[name]

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> Relay:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- internals ---------------------------------------------------------------

    def resolve_model(self, model: str, app: str | None = None) -> str:
        """The provider model a name becomes: [apps.<app>.models], then [models], else the name itself."""
        return self.config.route(model, app or self.app)[0]

    def _request(self, transport, messages, model, tools, files, params, app=None) -> tuple[ChatRequest, bool]:
        msgs = normalize_messages(messages, files)
        resolved, defaults = self.config.route(model, app or self.app)
        params = {**defaults, **params}  # config gives defaults (e.g. reasoning_effort); the call wins
        emulated = False
        if tools:
            tools = normalize_tools(tools)
            if transport.tools_mode == "emulated":
                msgs, tools, emulated = to_emulated(msgs, tools), None, True
        trace_id = uuid.uuid4().hex[:12]
        return ChatRequest(model=resolved, messages=msgs, tools=tools, params=params, trace_id=trace_id), emulated


def _finish(response: Response, req: ChatRequest, emulated: bool) -> Response:
    response.trace_id = req.trace_id
    if emulated:
        response.text, response.tool_calls = parse_emulated(response.text)
    bad = [tc for tc in response.tool_calls if tc.arguments is None]
    if bad:
        raise InvalidToolCall(
            f"Model called tool {bad[0].name or '?'} with arguments that are not a JSON object",
            raw=bad[0].raw_arguments, response=response,
        )
    return response


def _tag(error: ModelRelayError, req: ChatRequest, transport) -> None:
    """Adds the call's identity to an error raised anywhere below."""
    error.context.setdefault("trace_id", req.trace_id)
    error.context.setdefault("transport", transport.name)
    error.context.setdefault("model", req.model)


def _make_client(config: Config) -> httpx.Client:
    verify: bool | str = config.ca_bundle or config.verify_ssl
    if verify is False:
        warnings.warn("modelrelay: SSL verification is disabled (verify_ssl = false).", stacklevel=3)
    return httpx.Client(verify=verify, timeout=config.timeout_seconds, headers=config.headers)
