import json
import threading

import httpx
import pytest

from modelrelay.server import make_server
from tests.conftest import make_relay


@pytest.fixture
def served(mock):
    """A `modelrelay serve` in front of the mock provider. Yields its /v1 URL."""
    _, url = mock
    server = make_server(make_relay(url, models={"fast": "mock-fast"}), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def post(server, body, **kw):
    return httpx.post(f"{server.url}/chat/completions", json=body, timeout=10, **kw)


def test_chat_completion(served):
    r = post(served, {"model": "m", "messages": [{"role": "user", "content": "oi"}], "temperature": 0.1})
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "echo: oi"}
    assert data["usage"]["total_tokens"] == 15


def test_model_alias_resolved_by_config(served):
    r = post(served, {"model": "fast", "messages": [{"role": "user", "content": "oi"}]})
    assert r.json()["model"] == "mock-fast"


def test_images_come_back_as_data_urls(served):
    r = post(served, {"model": "mock-image", "messages": [{"role": "user", "content": "a cat"}]})
    images = r.json()["choices"][0]["message"]["images"]
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_tool_calls(served):
    tools = [{"type": "function", "function": {"name": "soma", "parameters": {"type": "object"}}}]
    r = post(served, {"model": "m", "tools": tools, "messages": [{"role": "user", "content": "use the tool"}]})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "soma"
    assert json.loads(call["function"]["arguments"]) == {"a": 2, "b": 3}


def test_stream_sse(served):
    body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hello world"}]}
    with httpx.stream("POST", f"{served.url}/chat/completions", json=body, timeout=10) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(x) for x in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert len(chunks) > 2 and text.strip() == "echo: hello world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_models_and_health(served):
    assert httpx.get(f"{served.url}/models").json()["data"] == [
        {"id": "fast", "object": "model", "owned_by": "modelrelay"}]
    assert httpx.get(served.url.replace("/v1", "/health")).json()["ok"] is True


def test_bad_requests(served):
    assert post(served, {"messages": []}).status_code == 400
    assert httpx.post(f"{served.url}/nope", json={}).status_code == 404
    r = httpx.post(f"{served.url}/chat/completions", content=b"{oops", timeout=10)
    assert r.status_code == 400


def test_upstream_error_is_reported(served):
    r = post(served, {"model": "mock-slow", "messages": [{"role": "user", "content": "oi"}]})
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["type"] == "ProviderError" and err["upstream_status"] == 504


def test_api_key_required_when_set(mock):
    _, url = mock
    server = make_server(make_relay(url), port=0, api_key="s3cret")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = {"model": "m", "messages": [{"role": "user", "content": "oi"}]}
        assert post(server, body).status_code == 401
        assert post(server, body, headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert httpx.get(server.url.replace("/v1", "/health")).status_code == 200  # health stays open
    finally:
        server.shutdown()
        server.server_close()
