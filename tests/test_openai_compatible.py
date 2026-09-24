import base64

import pytest

from modelrelay import PayloadTooLarge, ProviderError
from tests.conftest import make_relay

TOOLS = [{"name": "soma", "description": "Soma dois números", "parameters": {
    "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}]


def test_chat(mock):
    _, url = mock
    resp = make_relay(url).chat("oi", model="gpt-4o")
    assert resp.text == "echo: oi"
    assert resp.usage.input_tokens == 10


def test_model_map(mock):
    _, url = mock
    relay = make_relay(url, models={"gpt-4o": "region1;gpt-4o"})
    assert relay.chat("oi", model="gpt-4o").model == "region1;gpt-4o"
    assert relay.chat("oi", model="other").model == "other"


def test_history(mock):
    _, url = mock
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "echo: first"},
        {"role": "user", "content": "second"},
    ]
    assert make_relay(url).chat(msgs, model="m").text == "echo: second"


def test_stream_deltas(mock):
    _, url = mock
    relay = make_relay(url)
    assert relay.supports_text_stream
    events = list(relay.stream("um dois tres", model="m"))
    deltas = [e.text for e in events if e.type == "delta"]
    assert len(deltas) > 1
    assert events[-1].type == "done"
    assert events[-1].response.text.strip() == "echo: um dois tres"


def test_native_tools(mock):
    _, url = mock
    relay = make_relay(url)
    resp = relay.chat("use a tool", model="m", tools=TOOLS)
    assert resp.tool_calls[0].name == "soma"
    assert resp.tool_calls[0].arguments == {"a": 2, "b": 3}

    msgs = [{"role": "user", "content": "use a tool"}, resp.to_message(),
            {"role": "tool", "tool_call_id": resp.tool_calls[0].id, "content": "5"}]
    assert relay.chat(msgs, model="m", tools=TOOLS).text == "tool said: 5"


def test_native_tools_stream(mock):
    _, url = mock
    done = list(make_relay(url).stream("use a tool", model="m", tools=TOOLS))[-1]
    assert done.response.tool_calls[0].arguments == {"a": 2, "b": 3}


def test_image_output(mock, tmp_path):
    _, url = mock
    resp = make_relay(url).chat("draw a cat", model="mock-image")
    path = resp.images[0].save(tmp_path / "cat.png")
    assert path.read_bytes().startswith(b"\x89PNG")


def test_image_output_stream(mock):
    _, url = mock
    done = list(make_relay(url).stream("draw", model="mock-image"))[-1]
    assert done.response.images[0].mime_type == "image/png"


def test_files_are_attached(mock, tmp_path):
    _, url = mock
    img = tmp_path / "x.png"
    img.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    resp = make_relay(url).chat("what is this", model="m", files=[img, pdf])
    assert resp.text == "echo: what is this [2 attachment(s)]"


def test_payload_limit(mock, tmp_path):
    _, url = mock
    big = tmp_path / "big.pdf"
    big.write_bytes(b"0" * 900_000)
    with pytest.raises(PayloadTooLarge):
        make_relay(url, max_payload_mb=1).chat("x", model="m", files=[big])


def test_time_limit_error(mock):
    _, url = mock
    with pytest.raises(ProviderError) as err:
        make_relay(url).chat("x", model="mock-slow")
    assert err.value.status == 504


def test_bad_key_is_not_retried(mock):
    state, url = mock
    with pytest.raises(ProviderError) as err:
        make_relay(url, api_key="wrong").chat("x", model="m")
    assert err.value.status == 403
