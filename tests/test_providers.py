"""Several providers in one config: [providers.<name>] + `provider = "..."` on a model entry."""
import pytest

from modelrelay import Config, ConfigError, Relay
from modelrelay.testing import mock_server


@pytest.fixture
def two_mocks():
    """Two independent fake providers, each accepting only its own key."""
    a, state_a, url_a = mock_server.start()
    b, state_b, url_b = mock_server.start()
    state_a.static_keys = {"key-a"}
    state_b.static_keys = {"key-b"}
    yield (state_a, url_a), (state_b, url_b)
    a.shutdown()
    b.shutdown()


def providers_config(url_a, url_b, **extra):
    return Config.from_dict({
        "provider": "alpha",
        "providers": {
            "alpha": {"base_url": f"{url_a}/v1", "api_key": "key-a", "timeout_seconds": 5},
            "beta": {"base_url": f"{url_b}/v1", "api_key": "key-b", "timeout_seconds": 5},
        },
        "models": {
            "text": "model-a",                                                  # default provider
            "image": {"provider": "beta", "model": "mock-image"},
            "deep": {"provider": "beta", "model": "model-b", "reasoning_effort": "high"},
        },
        **extra,
    })


def test_each_role_goes_to_its_provider_with_its_own_key(two_mocks):
    (state_a, url_a), (state_b, url_b) = two_mocks
    relay = Relay(providers_config(url_a, url_b))
    assert relay.chat("oi", model="text").model == "model-a"
    assert state_a.last_headers["Authorization"] == "Bearer key-a"
    resp = relay.chat("um gato", model="image")
    assert resp.images and state_b.last_headers["Authorization"] == "Bearer key-b"


def test_provider_is_not_sent_as_a_param_but_other_defaults_are(two_mocks):
    (_, url_a), (state_b, url_b) = two_mocks
    config = providers_config(url_a, url_b)
    assert config.route("deep") == ("model-b", {"reasoning_effort": "high"})
    assert config.provider_for("deep") == "beta"
    assert config.provider_for("text") == "alpha"
    assert config.provider_for("not-an-alias") == "alpha"


def test_stream_uses_the_role_provider(two_mocks):
    (_, url_a), (state_b, url_b) = two_mocks
    relay = Relay(providers_config(url_a, url_b))
    events = list(relay.stream("olá", model="deep"))
    assert events[-1].response.text.strip() == "echo: olá"
    assert state_b.last_headers["Authorization"] == "Bearer key-b"


def test_apps_can_pick_another_provider(two_mocks):
    (state_a, url_a), (state_b, url_b) = two_mocks
    config = providers_config(url_a, url_b, apps={"wotan": {"models": {"text": {"provider": "beta", "model": "model-b"}}}})
    relay = Relay(config, app="wotan")
    assert relay.chat("oi", model="text").model == "model-b"
    assert state_b.last_headers["Authorization"] == "Bearer key-b"


def test_providers_do_not_inherit_each_others_credentials():
    config = Config.from_dict({"api_key": "top-secret", "providers": {"x": {"base_url": "http://x/v1"}}})
    assert config.for_provider("x").api_key is None


def test_old_single_provider_files_keep_working(mock):
    _, url = mock
    relay = Relay(Config.from_dict({"base_url": f"{url}/v1", "api_key": "test-key", "models": {"text": "m1"}}))
    assert relay.chat("oi", model="text").model == "m1"


@pytest.mark.parametrize("data, message", [
    ({"models": {"text": {"provider": "nope", "model": "m"}}}, 'provider "nope"'),
    ({"provider": "nope"}, 'provider "nope"'),
    ({"providers": {"x": {"models": {}}}}, r"\[providers.x\]"),
    ({"providers": {"x": "http://x"}}, r"\[providers.x\] must be a table"),
])
def test_bad_provider_configs_are_explained(data, message):
    with pytest.raises(ConfigError, match=message):
        Config.from_dict(data)
