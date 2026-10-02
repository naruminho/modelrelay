from __future__ import annotations

import json
import time
from typing import Iterator

from .._util import log
from ..errors import StreamInterrupted, UnexpectedResponse
from ..types import ChatRequest, Event, Response
from .base import Transport
from .openai_format import StreamAccumulator, parse_completion


class OpenAICompatible(Transport):
    """Any API that speaks POST /chat/completions: OpenAI, OpenRouter, LLM proxies..."""

    def build_body(self, req: ChatRequest, stream: bool) -> dict:
        payload = {"model": req.model, "messages": req.messages, **req.params}
        if req.tools:
            payload["tools"] = req.tools
        if stream:
            payload["stream"] = True
        return self.body_for(payload)

    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def complete(self, req: ChatRequest) -> Response:
        data = self.send("POST", self.url(), req, self.build_body(req, stream=False))
        return parse_completion(data)

    def stream(self, req: ChatRequest) -> Iterator[Event]:
        start = time.monotonic()
        acc = StreamAccumulator()
        finished = False
        last_alive = None

        def alive():
            # a reasoning model can think for minutes before the first piece of text: say it is still there
            # (the caller's idle timeout would give up), but not for every reasoning token
            nonlocal last_alive
            now = time.monotonic()
            if last_alive is None or now - last_alive >= 2:
                last_alive = now
                return Event("alive", elapsed=now - start, trace_id=req.trace_id)
            return None

        with self.open_stream(self.url(), req, self.build_body(req, stream=True)) as r:
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    if line.startswith(":") and (ev := alive()):  # SSE comments / keep-alives
                        yield ev
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as e:
                    raise UnexpectedResponse(f"Invalid JSON in stream: {data[:500]!r}", body=data,
                                             trace_id=req.trace_id) from e
                text = acc.add(chunk)
                if text:
                    yield Event("delta", text=text, elapsed=time.monotonic() - start, trace_id=req.trace_id)
                elif ev := alive():  # reasoning, empty deltas
                    yield ev

        response = acc.response()
        if not finished and not acc.finish_reason:
            log.warning("%s stream ended without [DONE] after %.1fs", req.trace_id, time.monotonic() - start)
            raise StreamInterrupted(
                "Stream ended before the provider finished (connection cut or server timeout)",
                partial=response, trace_id=req.trace_id, elapsed=round(time.monotonic() - start, 2),
                received_chars=len(response.text),
            )
        yield Event("done", response=response, elapsed=time.monotonic() - start, trace_id=req.trace_id)
