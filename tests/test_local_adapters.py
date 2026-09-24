"""Adapters that live as single files in ~/.modelrelay/adapters/."""

import pytest

from modelrelay import Config, ConfigError, Relay
from modelrelay.cli import main


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("MODELRELAY_CONFIG", raising=False)
    return tmp_path / ".modelrelay"


def write_adapter(home, name, body):
    (home / "adapters").mkdir(parents=True, exist_ok=True)
    (home / "adapters" / f"{name}.py").write_text(body)


def test_adapter_file_is_used_by_the_config(home, mock):
    _, url = mock
    write_adapter(home, "corp_test_a", "from modelrelay.testing.mock_adapter import MockJobsTransport as GatewayJobs\n")
    (home / "config.toml").write_text(
        f'base_url = "{url}/v1"\napi_key = "test-key"\ntransport = "corp_test_a:GatewayJobs"\n'
        '[transports."corp_test_a:GatewayJobs"]\npoll_interval = 0.01\n'
    )
    relay = Relay()
    assert relay.chat("oi", model="m").text == "echo: oi"
    assert type(relay.transport("corp_test_a:GatewayJobs")).__name__ == "MockJobsTransport"


def test_missing_class_is_explicit(home):
    write_adapter(home, "corp_test_b", "X = 1\n")
    relay = Relay(Config.from_dict({"transport": "corp_test_b:Nope", "api_key": "k"}))
    with pytest.raises(ConfigError, match="'Nope' not found"):
        relay.transport("corp_test_b:Nope")


def test_init_gateway_creates_config_and_adapter_skeleton(home):
    assert main(["init", "--template", "gateway"]) == 0
    adapter = home / "adapters" / "gateway.py"
    assert adapter.is_file() and "class GatewayJobs(JobsTransport)" in adapter.read_text()
    config = Config.load()
    assert config.transport == "gateway:GatewayJobs"
    assert "gateway:GatewayJobs" in config.transports

    # The skeleton loads, and every hook says what to fill in.
    relay = Relay(config)
    transport = relay.transport("gateway:GatewayJobs")
    with pytest.raises(NotImplementedError, match="TODO"):
        transport.build_submit(None)

    # A second init keeps the existing adapter.
    (home / "config.toml").unlink()
    adapter.write_text("# my work\n")
    assert main(["init", "--template", "gateway"]) == 0
    assert adapter.read_text() == "# my work\n"
