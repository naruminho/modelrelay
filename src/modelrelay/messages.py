from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def file_part(f) -> dict:
    """A local file (image or document) as an OpenAI-style content part."""
    if isinstance(f, dict):
        return f
    path = Path(f)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    if mime.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "file", "file": {"filename": path.name, "file_data": url}}


def normalize_messages(messages, files=None) -> list[dict]:
    """Accepts a plain string or a list of messages; attaches `files` to the last user message."""
    if isinstance(messages, str):
        msgs = [{"role": "user", "content": messages}]
    else:
        msgs = [dict(m) for m in messages]
    if not files:
        return msgs

    target = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
    if target is None:
        raise ValueError("`files` needs at least one user message to attach to.")
    content = target.get("content")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
    target["content"] = parts + [file_part(f) for f in files]
    return msgs


def text_of(content) -> str:
    """Text inside a message content (string or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""
