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


def test_credentials_in_config_file_win_over_env(mock, monkeypatch):
    state, url = mock
    monkeypatch.setenv("MODELRELAY_CLIENT_ID", "")
    monkeypatch.setenv("MODELRELAY_CLIENT_SECRET", "")
    relay = make_relay(url, auth="client_credentials",
                       auth_options=cc_options(url, client_id="file-id", client_secret="file-secret"))
    assert relay.auth.provider.client_id == "file-id"
    assert relay.chat("a", model="m").text == "echo: a"


def test_show_masks_credentials(tmp_path, monkeypatch, capsys):
    from modelrelay.cli import main
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("MODELRELAY_CONFIG", raising=False)
    (tmp_path / ".modelrelay").mkdir()
    (tmp_path / ".modelrelay" / "config.toml").write_text(
        'auth = "client_credentials"\n[auth_options]\ntoken_url = "http://x"\n'
        'client_id = "my-id"\nclient_secret = "my-secret"\n'
    )
    assert main(["show"]) == 0
    out = capsys.readouterr().out
    assert "my-id" not in out and "my-secret" not in out
