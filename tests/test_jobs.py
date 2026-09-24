import pytest

from modelrelay import JobFailed, JobTimeout
from tests.conftest import FAST_POLLING, JOBS, make_relay


def jobs_relay(url, **overrides):
    return make_relay(url, transport=JOBS, transports=FAST_POLLING, **overrides)


def test_chat_via_jobs(mock):
    _, url = mock
    assert jobs_relay(url).chat("oi", model="m").text == "echo: oi"


def test_stream_via_jobs_gives_status_events(mock):
    _, url = mock
    relay = jobs_relay(url)
    assert not relay.supports_text_stream
    events = list(relay.stream("oi", model="m"))
    assert [e.type for e in events] == ["queued", "queued", "running", "done"]
    assert events[-1].response.text == "echo: oi"
    assert events[-1].job_id


def test_stream_and_chat_can_use_different_transports(mock):
    _, url = mock
    relay = jobs_relay(url, stream_transport="openai_compatible")
    assert relay.supports_text_stream
    assert any(e.type == "delta" for e in relay.stream("oi", model="m"))


def test_tools_via_jobs(mock):
    _, url = mock
    resp = jobs_relay(url).chat("use a tool", model="m", tools=[{"name": "soma", "parameters": {}}])
    assert resp.tool_calls[0].name == "soma"


def test_failed_job(mock):
    _, url = mock
    with pytest.raises(JobFailed, match="model exploded"):
        jobs_relay(url).chat("oi", model="mock-fail")


def test_job_timeout(mock):
    _, url = mock
    options = {JOBS: {"poll_interval": 0.01, "max_wait_seconds": 0}}
    with pytest.raises(JobTimeout):
        make_relay(url, transport=JOBS, transports=options).chat("oi", model="m")
