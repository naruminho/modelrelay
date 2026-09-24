from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Iterator

import httpx

from .._util import log
from ..errors import PayloadTooLarge, ProviderError, UnexpectedResponse
from ..types import ChatRequest, Event, Response


class Transport:
    """How a request reaches the model.

    An adapter can subclass this directly and implement `complete` and `stream`
    any way it needs, or subclass JobsTransport / OpenAICompatible and override
    only their hooks. Relay only relies on these two methods and on
    `supports_text_stream`, so nothing about a gateway's contract is fixed here.

    Options (config `[transports.<name>]`) understood by every transport:
      base_url      overrides the global base_url
      tools_mode    "native" | "emulated", overrides the global one
      extra_body    dict merged into every request body
      extra_headers dict added to every request
    Everything else is left in `self.options` for the subclass.
    """

    supports_text_stream = True

    def __init__(self, http: httpx.Client, auth: httpx.Auth, config, options: dict | None = None):
        self.http = http
        self.auth = auth
        self.config = config
        self.options = dict(options or {})
        self.base_url = self.options.pop("base_url", config.base_url).rstrip("/")
        self.tools_mode = self.options.pop("tools_mode", config.tools_mode)
        self.extra_body = self.options.pop("extra_body", {})
        self.extra_headers = self.options.pop("extra_headers", {})

    @property
    def name(self) -> str:
        return type(self).__name__

    def complete(self, req: ChatRequest) -> Response:
        raise NotImplementedError

    def stream(self, req: ChatRequest) -> Iterator[Event]:
        raise NotImplementedError

    # ---- helpers for subclasses ------------------------------------------------------

    def headers_for(self, req: ChatRequest | None) -> dict:
        """Headers for one request. Override to add per-request headers."""
        return dict(self.extra_headers)

    def body_for(self, payload: dict) -> dict:
        """Final body: payload + extra_body, checked against max_payload_mb."""
        body = {**payload, **self.extra_body}
        self.check_size(body)
        return body

    def check_size(self, payload: dict) -> None:
        limit = self.config.max_payload_mb
        if not limit:
            return
        size = len(json.dumps(payload, ensure_ascii=False).encode())
        if size > limit * 1024 * 1024:
            raise PayloadTooLarge(
                f"Request has {size / 1024 / 1024:.1f} MB, above the {limit} MB limit "
                "(base64 files grow ~33%)",
                size_mb=round(size / 1024 / 1024, 2),
            )

    def send(self, method: str, url: str, req: ChatRequest | None = None, json_body: dict | None = None) -> dict:
        """One HTTP call returning parsed JSON. Every failure becomes a ProviderError with context."""
        trace_id = req.trace_id if req else None
        start = time.monotonic()
        log.debug("%s %s -> %s %s", trace_id, self.name, method, url)
        try:
            r = self.http.request(method, url, json=json_body, headers=self.headers_for(req), auth=self.auth)
        except httpx.HTTPError as e:
            log.warning("%s %s %s %s failed: %r", trace_id, self.name, method, url, e)
            raise ProviderError(
                f"Network error: {type(e).__name__}: {e}",
                method=method, url=url, trace_id=trace_id, elapsed=_secs(start),
            ) from e
        log.debug("%s %s <- %s %s %s (%.2fs)", trace_id, self.name, r.status_code, method, url, time.monotonic() - start)
        return json_or_raise(r, trace_id=trace_id, elapsed=_secs(start))

    @contextmanager
    def open_stream(self, url: str, req: ChatRequest, json_body: dict):
        """POST that keeps the response open for streaming; errors become ProviderError."""
        start = time.monotonic()
        log.debug("%s %s -> POST %s (stream)", req.trace_id, self.name, url)
        try:
            with self.http.stream("POST", url, json=json_body, headers=self.headers_for(req), auth=self.auth) as r:
                if r.status_code >= 400:
                    r.read()
                    json_or_raise(r, trace_id=req.trace_id, elapsed=_secs(start))
                yield r
        except httpx.HTTPError as e:
            log.warning("%s %s stream %s failed: %r", req.trace_id, self.name, url, e)
            raise ProviderError(
                f"Network error during stream: {type(e).__name__}: {e}",
                method="POST", url=url, trace_id=req.trace_id, elapsed=_secs(start),
            ) from e


def json_or_raise(r: httpx.Response, **context) -> dict:
    """Parses a JSON response. Raises ProviderError (HTTP >= 400 or an 'error' field) or
    UnexpectedResponse (not JSON). The full body is always kept in `error.body`."""
    context = {"method": r.request.method, "url": str(r.request.url), **context}
    try:
        data = r.json()
    except ValueError:
        data = None
    if r.status_code >= 400:
        detail = _error_message(data) or r.text[:500]
        raise ProviderError(f"HTTP {r.status_code}: {detail}", r.status_code, data if data is not None else r.text, **context)
    if data is None:
        raise UnexpectedResponse(f"Expected JSON, got: {r.text[:500]!r}", r.status_code, r.text, **context)
    if isinstance(data, dict) and data.get("error"):
        raise ProviderError(f"Provider error: {_error_message(data)}", r.status_code, data, **context)
    return data


def _error_message(data) -> str:
    if not isinstance(data, dict):
        return ""
    err = data.get("error")
    if isinstance(err, dict):
        message = str(err.get("message") or err)
        # Gateways like OpenRouter wrap the upstream provider's error; surface its reason too.
        upstream = _upstream_message((err.get("metadata") or {}).get("raw"))
        return f"{message} (upstream: {upstream})" if upstream and upstream not in message else message
    return str(err or data.get("message") or data.get("detail") or "")


def _upstream_message(raw) -> str:
    if not raw:
        return ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw[:300]
    if isinstance(raw, dict):
        inner = raw.get("error", raw)
        if isinstance(inner, dict):
            return str(inner.get("message") or "")[:300]
        return str(inner)[:300]
    return str(raw)[:300]


def _secs(start: float) -> float:
    return round(time.monotonic() - start, 2)
