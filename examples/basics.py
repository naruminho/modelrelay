"""Everything a project uses from modelrelay, once each. Runs the same at home and at work.

    python examples/basics.py
    python examples/basics.py --model gpt-4o --image-model gemini-2.5-flash-image

Uses ~/.modelrelay/config.toml. Model names are the ones in that file's [models].
"""

import argparse
import json
import tempfile
from pathlib import Path

from modelrelay import ModelRelayError, Relay

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt-4o-mini")
parser.add_argument("--image-model", default="gemini-2.5-flash-image")
args = parser.parse_args()

llm = Relay()
print(f"config: {llm.config.source}\n")


# 1. A simple question
resp = llm.chat("In one sentence, what is a relay?", model=args.model)
print("1.", resp.text)


# 2. A conversation: you keep the history and send it every time
history = [{"role": "system", "content": "You are concise."}]
for question in ["My name is Ana.", "What is my name?"]:
    history.append({"role": "user", "content": question})
    resp = llm.chat(history, model=args.model)
    history.append(resp.to_message())
    print("2.", question, "->", resp.text)


# 3. Streaming: text as it arrives, or status updates on job-based gateways
print("3. ", end="")
for ev in llm.stream("Count from 1 to 5.", model=args.model):
    if ev.type == "delta":
        print(ev.text, end="", flush=True)
    elif ev.type in ("queued", "running"):
        print(f"[{ev.type} {ev.elapsed:.0f}s] ", end="", flush=True)
    elif ev.type == "done":
        if not llm.supports_text_stream:
            print(ev.response.text, end="")
        print()


# 4. Tools: the model asks, your code runs the function, the model answers
def get_weather(city: str) -> str:
    return f"Sunny, 25 degrees Celsius in {city}"

tools = [{
    "name": "get_weather",
    "description": "Current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}]
messages = [{"role": "user", "content": "What's the weather in Lisbon?"}]
resp = llm.chat(messages, model=args.model, tools=tools)
while resp.tool_calls:
    messages.append(resp.to_message())
    for call in resp.tool_calls:
        result = get_weather(**call.arguments)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    resp = llm.chat(messages, model=args.model, tools=tools)
print("4.", resp.text)


# 5. Files in (images, PDFs) and images out
out = Path(tempfile.mkdtemp())
resp = llm.chat("Draw a small green circle on a white background.", model=args.image_model,
                modalities=["image", "text"])
if resp.images:
    image = resp.images[0].save(out / "circle.png")
    print("5. generated", image)
    resp = llm.chat("What shape and colour is in this image? Answer in 3 words.", model=args.image_model,
                    files=[image])
    print("5. described as:", resp.text)
else:
    print("5. no image returned:", resp.text)


# 6. Errors are exceptions with everything needed to trace them
try:
    llm.chat("hi", model="no-such-model")
except ModelRelayError as e:
    print("6.", type(e).__name__)
    print("   message:", e.message)
    print("   context:", json.dumps(e.context, default=str))
