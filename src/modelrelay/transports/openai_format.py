"""Parsing of OpenAI-style chat completion payloads (also used by OpenRouter and many gateways)."""

from __future__ import annotations

from .._util import log
from ..errors import ProviderError, UnexpectedResponse
from ..tools import make_tool_call
from ..types import Image, Response, Usage


def parse_completion(data: dict) -> Response:
    """OpenAI chat completion -> Response. Raises UnexpectedResponse if the shape is wrong."""
    if not isinstance(data, dict):
        raise UnexpectedResponse(f"Expected a chat completion object, got {type(data).__name__}", body=data)
    if data.get("error"):
        raise ProviderError(f"Provider error: {data['error']}", body=data)
    choices = data.get("choices") or []
    if not choices:
        raise UnexpectedResponse("Chat completion has no 'choices'", body=data)
    choice = choices[0]
    msg = choice.get("message") or {}
    text, images = _content(msg.get("content"))
    images += _images(msg.get("images"))
    tool_calls = [
        make_tool_call(
            tc.get("id") or f"call_{i}",
            (tc.get("function") or {}).get("name", ""),
            (tc.get("function") or {}).get("arguments"),
        )
        for i, tc in enumerate(msg.get("tool_calls") or [])
    ]
    return Response(
        text=text,
        tool_calls=tool_calls,
        images=images,
        usage=_usage(data.get("usage")),
        finish_reason=choice.get("finish_reason"),
        model=data.get("model"),
        raw=data,
    )


class StreamAccumulator:
    """Builds a Response out of streamed chunks."""

    def __init__(self):
        self.text: list[str] = []
        self.tool_calls: dict[int, dict] = {}
        self.images: list[Image] = []
        self.finish_reason = None
        self.usage = None
        self.model = None

    def add(self, chunk: dict) -> str:
        """Adds one chunk; returns the new text in it (may be empty)."""
        if chunk.get("error"):
            raise ProviderError(f"Provider error inside the stream: {chunk['error']}", body=chunk)
        self.model = chunk.get("model") or self.model
        if chunk.get("usage"):
            self.usage = _usage(chunk["usage"])
        new_text = ""
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text, images = _content(delta.get("content"))
            new_text += text
            self.images += images + _images(delta.get("images"))
            for tc in delta.get("tool_calls") or []:
                entry = self.tool_calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                entry["id"] = tc.get("id") or entry["id"]
                fn = tc.get("function") or {}
                entry["name"] = fn.get("name") or entry["name"]
                entry["arguments"] += fn.get("arguments") or ""
            self.finish_reason = choice.get("finish_reason") or self.finish_reason
        self.text.append(new_text)
        return new_text

    def response(self) -> Response:
        return Response(
            text="".join(self.text),
            tool_calls=[
                make_tool_call(e["id"] or f"call_{i}", e["name"], e["arguments"])
                for i, e in sorted(self.tool_calls.items())
            ],
            images=self.images,
            usage=self.usage,
            finish_reason=self.finish_reason,
            model=self.model,
        )


def _content(content) -> tuple[str, list[Image]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    text, images = [], []
    for part in content:
        kind = part.get("type")
        if kind in ("text", "output_text"):
            text.append(part.get("text", ""))
        elif kind in ("image_url", "output_image"):
            url = part.get("image_url", {}).get("url") if kind == "image_url" else part.get("image_url") or part.get("url")
            if url:
                images.append(Image.from_url(url))
        else:
            log.warning("ignoring unknown content part type %r (still available in response.raw)", kind)
    return "".join(text), images


def _images(items) -> list[Image]:
    out = []
    for item in items or []:
        url = (item.get("image_url") or {}).get("url") if isinstance(item, dict) else None
        if url:
            out.append(Image.from_url(url))
    return out


def _usage(u) -> Usage | None:
    if not u:
        return None
    return Usage(
        input_tokens=u.get("prompt_tokens", u.get("input_tokens")),
        output_tokens=u.get("completion_tokens", u.get("output_tokens")),
    )
