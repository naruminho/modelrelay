import time

import httpx
import pytest

from modelrelay import AuthError, ClientCredentials, ConfigError, StaticToken
from modelrelay.testing import mock_server
from tests.conftest import make_relay


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def cc_options(url, **extra):
    return {"token_url": f"{url}/auth/token", "token_field": "data.token", **extra}


def test_token_is_reused_then_renewed_before_expiry(mock):
    state, url = mock
    clock = FakeClock()
    provider = ClientCredentials(httpx.Client(), client_id="id", client_secret="secret",
                                 clock=clock, **cc_options(url))
    first = provider.get_token()
    clock.now = 27 * 60
    assert provider.get_token() == first
    clock.now = 28 * 60 + 1  # inside the 2-minute margin
    assert provider.get_token() != first
    assert state.token_requests == 2


def test_form_request_format(mock):
    _, url = mock
    provider = ClientCredentials(httpx.Client(), client_id="id", client_secret="s",
                                 request_format="form", **cc_options(url))
    assert provider.get_token().startswith("mock-")


def test_client_credentials_end_to_end(mock, credentials):
    state, url = mock
    relay = make_relay(url, auth="client_credentials", auth_options=cc_options(url))
    assert relay.chat("a", model="m").text == "echo: a"
    assert relay.chat("b", model="m").text == "echo: b"
    assert state.token_requests == 1


def test_expired_token_is_renewed_on_403():
    # The server kills tokens after 0.2s, but the client thinks they last 30 min.
    server, state, url = mock_server.start(token_ttl=0.2)
    try:
        relay = make_relay(url, auth="client_credentials", auth_options=cc_options(url))
        relay.auth.provider.client_id, relay.auth.provider.client_secret = "id", "secret"
        relay.chat("a", model="m")
        time.sleep(0.3)
        assert relay.chat("b", model="m").text == "echo: b"
        assert state.token_requests == 2
    finally:
        server.shutdown()


def test_bad_token_endpoint(mock):
    _, url = mock
    provider = ClientCredentials(httpx.Client(), client_id="id", client_secret="",
                                 **cc_options(url))
    with pytest.raises(ConfigError):
        provider.get_token()
    provider.client_secret = "s"
    provider.token_field = "nope"
    with pytest.raises(AuthError):
        provider.get_token()


def test_missing_static_key_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        StaticToken(None, missing_hint="Set the OPENAI_API_KEY environment variable.").get_token()
