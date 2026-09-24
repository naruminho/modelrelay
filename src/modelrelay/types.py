from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None  # None only when raw_arguments is not valid JSON (see InvalidToolCall)
    raw_arguments: str = ""

    def to_openai(self) -> dict:
        args = self.raw_arguments if self.arguments is None else json.dumps(self.arguments, ensure_ascii=False)
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": args}}


@dataclass
class Image:
    data: bytes = b""
    mime_type: str = "image/png"
    url: str | None = None  # set when the provider returns a link instead of bytes

    @classmethod
    def from_url(cls, url: str) -> Image:
        if not url.startswith("data:"):
            return cls(url=url)
        header, b64 = url.split(",", 1)
        mime = header[5:].split(";")[0] or "image/png"
        return cls(data=base64.b64decode(b64), mime_type=mime)

    def save(self, path: str | Path) -> Path:
        if not self.data:
            raise ValueError(f"Image has no bytes, only a URL: {self.url}")
        path = Path(path)
        path.write_bytes(self.data)
        return path


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class Response:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    usage: Usage | None = None
    finish_reason: str | None = None
    model: str | None = None
    raw: object = None
    trace_id: str | None = None

    def to_message(self) -> dict:
        """The assistant message to append to your history."""
        msg: dict = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg


@dataclass
class Event:
    """Something that happened while waiting for a response.

    type is one of:
      queued  - the job was accepted (job-based transports only)
      running - still working; `elapsed` tells for how long
      delta   - a piece of text (only when the transport streams text)
      done    - finished; `response` holds the full Response
    """

    type: str
    text: str = ""
    elapsed: float = 0.0
    response: Response | None = None
    job_id: str | None = None
    trace_id: str | None = None


@dataclass
class ChatRequest:
    """What a transport receives. `model` is already resolved through the model map."""

    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    params: dict = field(default_factory=dict)
    trace_id: str = ""
