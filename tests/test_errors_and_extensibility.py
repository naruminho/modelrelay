"""Errors must never be swallowed, and adapters must be able to change anything."""

import logging

import pytest

from modelrelay import (
    ChatRequest,
    Event,
    InvalidToolCall,
    JobFailed,
    ProviderError,
    Response,
    StreamInterrupted,
    Transport,
    UnexpectedResponse,
)
from modelrelay.testing.mock_adapter import MockJobsTransport
from tests.conftest import FAST_POLLING, JOBS, make_relay

TOOLS = [{"name": "soma", "parameters": {"type": "object"}}]


def jobs_relay(url, **overrides):
    return make_relay(url, transport=JOBS, transports=FAST_POLLING, **overrides)


# ---- nothing disappears ----------------------------------------------------------------

def test_cut_stream_raises_with_partial_text(mock):
    _, url = mock
    events = []
    with pytest.raises(StreamInterrupted) as err:
        for ev in make_relay(url).stream("x", model="mock-cut"):
            events.append(ev)
    assert [e.type for e in events] == ["delta"]
    assert err.value.partial.text == "this answer gets cut "
    assert err.value.trace_id


def test_http_errors_carry_url_status_body_and_trace(mock):
    _, url = mock
    with pytest.raises(ProviderError) as err:
        make_relay(url).chat("x", model="mock-slow")
    e = err.value
    assert e.status == 504
    assert e.body == {"error": {"message": "upstream request timeout"}}
    assert e.context["url"].endswith("/v1/chat/completions")
    assert e.context["method"] == "POST"
    assert e.context["transport"] == "OpenAICompatible"
    assert e.context["model"] == "mock-slow"
    assert e.trace_id and e.trace_id in str(e)


def test_network_errors_are_provider_errors():
    relay = make_relay("http://127.0.0.1:9")  # nothing listens there
    with pytest.raises(ProviderError, match="Network error") as err:
        relay.chat("x", model="m")
    assert err.value.__cause__ is not None  # original exception kept


def test_invalid_tool_arguments_raise(mock):
    _, url = mock
    with pytest.raises(InvalidToolCall) as err:
        make_relay(url).chat("use a tool", model="mock-badtool", tools=TOOLS)
    assert err.value.raw == "{not json"
    assert err.value.response.tool_calls[0].name == "soma"


def test_invalid_emulated_tool_block_raises():
    from modelrelay.relay import _finish
    resp = Response(text='<tool_call>{"name": "soma", "arguments": </tool_call>')
    with pytest.raises(InvalidToolCall):
        _finish(resp, ChatRequest(model="m", messages=[], trace_id="t"), emulated=True)


def test_unknown_job_state_raises_instead_of_waiting_forever(mock):
    _, url = mock
    with pytest.raises(UnexpectedResponse, match="PAUSED") as err:
        jobs_relay(url).chat("x", model="mock-weird")
    assert err.value.body["data"]["execution"]["state"] == "PAUSED"
    assert err.value.context["job_id"]


def test_missing_field_in_contract_raises(mock):
    _, url = mock

    class Broken(MockJobsTransport):
        def parse_submit(self, data):
            from modelrelay import require
            return require(data, "data.execution.uuid", "execution id")

    relay = jobs_relay(url)
    relay._transports[JOBS] = Broken(relay.http, relay.auth, relay.config, FAST_POLLING[JOBS])
    with pytest.raises(UnexpectedResponse, match="data.execution.uuid"):
        relay.chat("x", model="m")


def test_failed_job_keeps_payload(mock):
    _, url = mock
    with pytest.raises(JobFailed) as err:
        jobs_relay(url).chat("x", model="mock-fail")
    assert err.value.body["data"]["output"]["error"]["message"] == "model exploded"


def test_poll_retries_are_logged(mock, caplog):
    _, url = mock
    with caplog.at_level(logging.WARNING, logger="modelrelay"):
        assert jobs_relay(url).chat("oi", model="mock-flaky").text == "echo: oi"
    assert any("poll failed (1/5)" in r.getMessage() for r in caplog.records)


def test_trace_header_is_sent(mock):
    state, url = mock
    resp = make_relay(url, trace_header="X-Request-ID").chat("x", model="m")
    assert state.last_headers["X-Request-ID"] == resp.trace_id


# ---- adapters can change anything -------------------------------------------------------

def test_extra_body_and_headers(mock):
    state, url = mock
    relay = make_relay(url, transports={"openai_compatible": {
        "extra_headers": {"X-App": "my-app"}, "extra_body": {"user": "me"}}})
    relay.chat("x", model="m")
    assert state.last_headers["X-App"] == "my-app"


def test_fully_custom_transport(mock):
    _, url = mock

    class Custom(Transport):
        """A transport written from scratch: Relay needs nothing else."""
        supports_text_stream = False

        def complete(self, req):
            return Response(text=f"custom:{req.model}:{req.messages[-1]['content']}")

        def stream(self, req):
            yield Event("running", elapsed=0.1)
            yield Event("done", response=self.complete(req))

    relay = make_relay(url, transport="custom", models={"gpt-4o": "x;gpt-4o"})
    relay._transports["custom"] = Custom(relay.http, relay.auth, relay.config)
    assert relay.chat("hi", model="gpt-4o").text == "custom:x;gpt-4o:hi"
    events = list(relay.stream("hi", model="gpt-4o"))
    assert [e.type for e in events] == ["running", "done"]
    assert all(e.trace_id for e in events)
