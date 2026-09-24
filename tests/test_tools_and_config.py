import pytest

from modelrelay import Config, ConfigError, Relay
from modelrelay.tools import normalize_tools, parse_emulated, to_emulated
from tests.conftest import make_relay

TOOLS = normalize_tools([{"name": "soma", "parameters": {"type": "object"}}])


def test_parse_emulated_calls():
    text = 'Vou somar.\n<tool_call>{"name": "soma", "arguments": {"a": 1, "b": {"c": 2}}}</tool_call>'
    rest, calls = parse_emulated(text)
    assert rest == "Vou somar."
    assert calls[0].name == "soma"
    assert calls[0].arguments == {"a": 1, "b": {"c": 2}}


def test_parse_emulated_without_calls():
    assert parse_emulated("just text") == ("just text", [])


def test_to_emulated_rewrites_history():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "soma 1 e 2"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "soma", "arguments": '{"a": 1, "b": 2}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "3"},
    ]
    out = to_emulated(msgs, TOOLS)
    assert out[0]["role"] == "system" and "<tool_call>" in out[0]["content"] and out[0]["content"].startswith("sys")
    assert out[2] == {"role": "assistant", "content": '<tool_call>{"name": "soma", "arguments": {"a": 1, "b": 2}}</tool_call>'}
    assert out[3]["role"] == "user" and "<tool_result" in out[3]["content"]
    assert all("tool_calls" not in m for m in out)


def test_emulated_mode_sends_no_tools_field(mock):
    _, url = mock
    # The mock only calls tools natively, so in emulated mode it just echoes.
    resp = make_relay(url, tools_mode="emulated").chat("use a tool", model="m", tools=TOOLS)
    assert resp.tool_calls == []
    assert resp.text == "echo: use a tool"


def test_config_file(tmp_path, monkeypatch):
    path = tmp_path / "modelrelay.toml"
    path.write_text(
        'base_url = "http://x/v1"\n'
        'transport = "jobs"\n'
        '[models]\n"gpt-4o" = "r1;gpt-4o"\n'
        '[transports.jobs]\npoll_interval = 2\n'
    )
    monkeypatch.setenv("MODELRELAY_CONFIG", str(path))
    config = Config.load()
    assert config.models == {"gpt-4o": "r1;gpt-4o"}
    assert Relay(config).transport("jobs").poll_interval == 2


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="bogus"):
        Config.from_dict({"bogus": 1})


def test_unknown_transport():
    with pytest.raises(ConfigError, match="nope"):
        Relay(Config.from_dict({"transport": "nope"})).transport("nope")


def test_disabling_ssl_warns():  # noqa: D103
    with pytest.warns(UserWarning, match="SSL"):
        Relay(Config.from_dict({"verify_ssl": False}))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("MODELRELAY_CONFIG", raising=False)
    (tmp_path / ".modelrelay").mkdir()
    return tmp_path / ".modelrelay"


def test_default_config_in_home_folder(home):
    (home / "config.toml").write_text('base_url = "http://home/v1"\n')
    config = Config.load()
    assert config.base_url == "http://home/v1"
    assert config.source == home / "config.toml"


def test_profiles(home):
    (home / "config.toml").write_text('base_url = "http://default/v1"\n')
    (home / "proxy.toml").write_text('base_url = "http://proxy/v1"\n')
    assert Relay(profile="proxy").config.base_url == "http://proxy/v1"
    with pytest.raises(ConfigError, match="Profile 'nope' not found"):
        Config.load(profile="nope")


def test_no_config_is_an_error(home):
    with pytest.raises(ConfigError, match="modelrelay init"):
        Config.load()
    with pytest.raises(ConfigError, match="init --profile proxy"):
        Relay(profile="proxy")


def test_cli_init_and_show(home, capsys):
    from modelrelay.cli import main
    assert main(["init"]) == 0
    assert (home / "config.toml").read_text().startswith("# modelrelay config: OpenRouter")
    assert main(["init"]) == 1  # never overwrites
    assert main(["init", "--profile", "work", "--template", "gateway"]) == 0
    work = Config.load(profile="work")
    assert work.auth == "client_credentials"
    assert work.transport == "gateway:GatewayJobs" and work.stream_transport == "openai_compatible"

    (home / "config.toml").write_text('api_key = "sk-secret"\n')
    assert main(["show"]) == 0
    out = capsys.readouterr().out
    assert "sk-secret" not in out and "api_key = ***" in out
    assert main(["show", "--profile", "missing"]) == 1


def test_invalid_toml_is_explicit(home):
    (home / "config.toml").write_text("base_url = \n")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        Config.load()


def test_source_cannot_be_set_from_file(home):
    (home / "config.toml").write_text('source = "x"\n')
    with pytest.raises(ConfigError, match="source"):
        Config.load()
