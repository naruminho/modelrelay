"""A model name can carry default request params: "text" = { model = "...", reasoning_effort = "high" }.
Works in [models] and [apps.<app>.models]; params sent by the app on a call win over the config."""
import threading

import httpx
import pytest

from modelrelay import Config, ConfigError
from modelrelay.cli import TEMPLATES, main
from modelrelay.server import APP_HEADER, make_server
from tests.conftest import make_relay

MODELS = {
    "text": {"model": "mock-text", "reasoning_effort": "medium"},
    "image": "mock-image",
}
APPS = {"wotan": {"models": {"text": {"model": "mock-strong", "reasoning_effort": "high", "temperature": 0.2}}}}


def test_route_splits_model_and_default_params():
    config = Config.from_dict({"models": MODELS, "apps": APPS})
    assert config.route("text") == ("mock-text", {"reasoning_effort": "medium"})
    assert config.route("image") == ("mock-image", {})
    assert config.route("text", "wotan") == ("mock-strong", {"reasoning_effort": "high", "temperature": 0.2})
    assert config.route("not-an-alias") == ("not-an-alias", {})
    assert config.models_for("wotan")["text"] == "mock-strong"  # names -> provider model, as before


@pytest.mark.parametrize("models, message", [
    ({"text": {"reasoning_effort": "high"}}, "needs a `model`"),
    ({"text": {"model": 3}}, "needs a `model`"),
    ({"text": 3}, "must be a model string or a table"),
])
def test_bad_model_entries_are_explained(models, message):
    with pytest.raises(ConfigError, match=message):
        Config.from_dict({"models": models})


def test_bad_app_model_entries_are_explained():
    with pytest.raises(ConfigError, match=r"\[apps.wotan.models\]"):
        Config.from_dict({"apps": {"wotan": {"models": {"text": {"effort": "x"}}}}})


class Capture:
    """Records the body the provider receives (wraps the transport)."""

    def __init__(self, relay):
        self.bodies = []
        transport = relay.transport(relay.config.transport)
        original = transport.complete

        def complete(req):
            self.bodies.append({"model": req.model, **req.params})
            return original(req)

        transport.complete = complete


def test_config_params_reach_the_provider_and_the_app_can_override(mock):
    _, url = mock
    relay = make_relay(url, models=MODELS, apps=APPS)
    cap = Capture(relay)
    relay.chat("hi", model="text")
    relay.chat("hi", model="text", app="wotan")
    relay.chat("hi", model="text", app="wotan", reasoning_effort="low")   # explicit on the call wins
    relay.chat("hi", model="image")
    assert cap.bodies == [
        {"model": "mock-text", "reasoning_effort": "medium"},
        {"model": "mock-strong", "reasoning_effort": "high", "temperature": 0.2},
        {"model": "mock-strong", "reasoning_effort": "low", "temperature": 0.2},
        {"model": "mock-image"},
    ]


def test_serve_applies_the_params_of_the_app(mock):
    _, url = mock
    relay = make_relay(url, models=MODELS, apps=APPS)
    cap = Capture(relay)
    server = make_server(relay, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = {"model": "text", "messages": [{"role": "user", "content": "oi"}]}
        httpx.post(f"{server.url}/chat/completions", json=body, headers={APP_HEADER: "wotan"}, timeout=10)
        httpx.post(f"{server.url}/chat/completions", json={**body, "reasoning_effort": "low"}, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
    assert cap.bodies == [
        {"model": "mock-strong", "reasoning_effort": "high", "temperature": 0.2},
        {"model": "mock-text", "reasoning_effort": "low"},
    ]


def test_show_prints_the_params(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[models]\n"text" = { model = "a", reasoning_effort = "high" }\n"image" = "b"\n', encoding="utf-8")
    monkeypatch.setenv("MODELRELAY_CONFIG", str(cfg))
    assert main(["show", "--app", "x"]) == 0
    out = capsys.readouterr().out
    assert "text = a  (reasoning_effort=high)" in out
    assert "image = b" in out


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_templates_explain_roles_and_reasoning(name):
    text = TEMPLATES[name]
    assert "reasoning_effort" in text
    assert "reads" in text and "generates" in text   # what each role must support
