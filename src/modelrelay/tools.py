"""Tool calling helpers, including emulation for providers without native support."""

from __future__ import annotations

import json
import re
import uuid

from .messages import text_of
from .types import ToolCall

# Captures anything between the tags, so a malformed block is reported instead of ignored.
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def normalize_tools(tools: list[dict]) -> list[dict]:
    """Accepts OpenAI format or the short {name, description, parameters} form."""
    out = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            out.append(t)
        else:
            out.append({"type": "function", "function": t})
    return out


def parse_arguments(raw) -> dict | None:
    """JSON arguments -> dict. Returns None when they are not a valid JSON object;
    Relay then raises InvalidToolCall, so nothing is silently dropped."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def make_tool_call(id: str, name: str, raw_arguments) -> ToolCall:
    raw = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments or {}, ensure_ascii=False)
    return ToolCall(id=id, name=name, arguments=parse_arguments(raw_arguments), raw_arguments=raw)


def emulation_prompt(tools: list[dict]) -> str:
    specs = [t["function"] for t in tools]
    return (
        "You can use tools. Available tools, described with JSON Schema:\n"
        f"{json.dumps(specs, ensure_ascii=False, indent=2)}\n\n"
        "To call a tool, reply with one or more blocks exactly like this and nothing else:\n"
        '<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>\n'
        "Tool results come back inside <tool_result> blocks. "
        "When no tool is needed, answer normally."
    )


def to_emulated(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Rewrites a conversation so tools live in the prompt instead of the API fields."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            blocks = [
                "<tool_call>"
                + json.dumps(
                    {"name": tc["function"]["name"], "arguments": parse_arguments(tc["function"].get("arguments"))},
                    ensure_ascii=False,
                )
                + "</tool_call>"
                for tc in m["tool_calls"]
            ]
            text = "\n".join(filter(None, [text_of(m.get("content")), *blocks]))
            out.append({"role": "assistant", "content": text})
        elif role == "tool":
            result = text_of(m.get("content"))
            out.append({
                "role": "user",
                "content": f'<tool_result id="{m.get("tool_call_id", "")}">\n{result}\n</tool_result>',
            })
        else:
            out.append(m)

    prompt = emulation_prompt(tools)
    if out and out[0].get("role") == "system" and isinstance(out[0].get("content"), str):
        out[0] = {**out[0], "content": out[0]["content"] + "\n\n" + prompt}
    else:
        out.insert(0, {"role": "system", "content": prompt})
    return out


def parse_emulated(text: str) -> tuple[str, list[ToolCall]]:
    """Extracts <tool_call> blocks. Returns (remaining text, tool calls)."""
    calls = []
    for match in _TOOL_CALL.finditer(text):
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            # Kept (with arguments=None) so Relay raises InvalidToolCall instead of dropping it.
            calls.append(ToolCall(id=call_id, name="", arguments=None, raw_arguments=match.group(1)))
            continue
        calls.append(make_tool_call(call_id, str(data.get("name", "")), data.get("arguments")))
    if not calls:
        return text, []
    return _TOOL_CALL.sub("", text).strip(), calls
