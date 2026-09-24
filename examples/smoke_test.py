"""Calls a real provider once per feature and prints what happened.

    python examples/smoke_test.py                              # uses examples/openrouter.toml
    python examples/smoke_test.py --profile config             # uses ~/.modelrelay/config.toml
    python examples/smoke_test.py path/to/config.toml --text-model X --image-model Y

Costs a few cents. Every check prints OK or FAIL with the full error.
"""

from __future__ import annotations

import argparse
import struct
import tempfile
import traceback
import zlib
from pathlib import Path

from modelrelay import InvalidToolCall, ModelRelayError, Relay

HERE = Path(__file__).parent
TOOLS = [{
    "name": "add",
    "description": "Adds two numbers",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    },
}]


def solid_png(size: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    row = b"\x00" + bytes(rgb) * size
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size))
            + chunk(b"IEND", b""))


MINI_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 100]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 55>>stream
BT /F1 18 Tf 20 50 Td (The secret word is PINEAPPLE) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
trailer<</Root 1 0 R>>
%%EOF"""

results: list[tuple[str, bool]] = []


def check(name):
    def wrap(fn):
        print(f"\n=== {name}")
        try:
            fn()
            print("OK")
            results.append((name, True))
        except Exception as e:  # noqa: BLE001 - we want to see everything here
            print(f"FAIL: {type(e).__name__}: {e}")
            if not isinstance(e, ModelRelayError):
                traceback.print_exc()
            elif getattr(e, "body", None) is not None:
                print(f"body: {e.body}")
            results.append((name, False))
        return fn
    return wrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(HERE / "openrouter.toml"))
    parser.add_argument("--profile", help="use ~/.modelrelay/<profile>.toml instead of a path")
    parser.add_argument("--text-model", default="gpt-4o-mini")
    parser.add_argument("--vision-model", default="gemini-2.5-flash")
    parser.add_argument("--image-model", default="gemini-2.5-flash-image")
    args = parser.parse_args()
    source = {"profile": args.profile} if args.profile else {"config_path": args.config}
    llm = Relay(**source)
    print(f"config: {llm.config.source}")
    tmp = Path(tempfile.mkdtemp())

    @check("chat")
    def _():
        r = llm.chat("Reply with exactly: pong", model=args.text_model, max_tokens=10)
        print(f"text={r.text!r} usage={r.usage} trace_id={r.trace_id}")
        assert "pong" in r.text.lower()

    @check("history")
    def _():
        msgs = [
            {"role": "system", "content": "Answer with one word."},
            {"role": "user", "content": "My favourite colour is teal. Remember it."},
            {"role": "assistant", "content": "Okay."},
            {"role": "user", "content": "What is my favourite colour?"},
        ]
        r = llm.chat(msgs, model=args.text_model, max_tokens=10)
        print(f"text={r.text!r}")
        assert "teal" in r.text.lower()

    @check("stream")
    def _():
        deltas, done = [], None
        for ev in llm.stream("Count from 1 to 10, separated by spaces.", model=args.text_model):
            if ev.type == "delta":
                deltas.append(ev.text)
            elif ev.type == "done":
                done = ev.response
        print(f"{len(deltas)} deltas, final={done.text!r}, finish={done.finish_reason}")
        assert len(deltas) > 1 and "10" in done.text

    @check("tools (native)")
    def _():
        msgs = [{"role": "user", "content": "What is 17 + 25? Use the add tool."}]
        r = llm.chat(msgs, model=args.text_model, tools=TOOLS)
        print(f"tool_calls={r.tool_calls}")
        call = r.tool_calls[0]
        assert call.name == "add"
        result = call.arguments["a"] + call.arguments["b"]
        msgs += [r.to_message(), {"role": "tool", "tool_call_id": call.id, "content": str(result)}]
        final = llm.chat(msgs, model=args.text_model, tools=TOOLS)
        print(f"final={final.text!r}")
        assert "42" in final.text

    @check("tools (native, stream)")
    def _():
        msgs = [{"role": "user", "content": "What is 17 + 25? Use the add tool."}]
        done = list(llm.stream(msgs, model=args.text_model, tools=TOOLS))[-1].response
        print(f"tool_calls={done.tool_calls}")
        assert done.tool_calls and done.tool_calls[0].name == "add"

    @check("tools (emulated)")
    def _():
        emu = Relay(**source, tools_mode="emulated")
        msgs = [{"role": "user", "content": "What is 17 + 25? Use the add tool."}]
        try:
            r = emu.chat(msgs, model=args.text_model, tools=TOOLS)
        except InvalidToolCall as e:
            print(f"raw={e.raw!r}")
            raise
        print(f"tool_calls={r.tool_calls} text={r.text!r}")
        call = r.tool_calls[0]
        msgs += [r.to_message(), {"role": "tool", "tool_call_id": call.id, "content": "42"}]
        final = emu.chat(msgs, model=args.text_model, tools=TOOLS)
        print(f"final={final.text!r}")
        assert "42" in final.text

    @check("image input")
    def _():
        png = tmp / "red.png"
        png.write_bytes(solid_png(64, (220, 20, 20)))
        r = llm.chat("What colour is this image? One word.", model=args.vision_model, files=[png])
        print(f"text={r.text!r}")
        assert "red" in r.text.lower()

    @check("pdf input")
    def _():
        pdf = tmp / "secret.pdf"
        pdf.write_bytes(MINI_PDF)
        r = llm.chat("What is the secret word in this PDF? One word.", model=args.vision_model, files=[pdf])
        print(f"text={r.text!r}")
        assert "pineapple" in r.text.lower()

    @check("image generation")
    def _():
        r = llm.chat("Generate a simple image of a blue square on a white background.",
                     model=args.image_model, modalities=["image", "text"])
        print(f"text={r.text[:80]!r} images={len(r.images)}")
        path = r.images[0].save(tmp / "generated.png")
        print(f"saved {path} ({path.stat().st_size} bytes)")

    @check("error is explicit (unknown model)")
    def _():
        try:
            llm.chat("hi", model="this-model/does-not-exist")
        except ModelRelayError as e:
            print(f"raised {type(e).__name__}: {e}")
            assert e.trace_id and "url" in e.context
            return
        raise AssertionError("no error raised")

    print("\n" + "\n".join(f"{'OK  ' if ok else 'FAIL'} {name}" for name, ok in results))


if __name__ == "__main__":
    main()
