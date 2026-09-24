import pytest

from modelrelay import Config, Relay
from modelrelay.testing import mock_server


@pytest.fixture
def mock():
    server, state, url = mock_server.start(request_limit=0.2, polls_to_finish=3)
    yield state, url
    server.shutdown()


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv("MODELRELAY_CLIENT_ID", "00000000-0000-0000-0000-000000000001")
    monkeypatch.setenv("MODELRELAY_CLIENT_SECRET", "00000000-0000-0000-0000-000000000002")


def make_relay(url, **overrides) -> Relay:
    data = {
        "base_url": f"{url}/v1",
        "auth": "static",
        "api_key": "test-key",
        "timeout_seconds": 5,
        **overrides,
    }
    return Relay(Config.from_dict(data))


JOBS = "modelrelay.testing.mock_adapter:MockJobsTransport"
FAST_POLLING = {JOBS: {"poll_interval": 0.01, "poll_max_interval": 0.02}}
