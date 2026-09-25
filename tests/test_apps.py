"""Per-app models: [apps.<app>.models] overrides [models] for one app; provider and auth stay shared."""
import threading
import tomllib

import httpx
import pytest

from modelrelay import Config, ConfigError
from modelrelay.cli import TEMPLATES, main
from modelrelay.server import APP_HEADER, make_server
from tests.conftest import make_relay

MODELS = {"text": "mock-text", "image": "mock-image"}
APPS = {"wotan": {"models": {"text": "mock-strong"}}, "sagadeck": {"models": {"image": "mock-image-pro"}}}


def test_models_for_merges_app_over_shared():
    config = Config.from_dict({"models": MODELS, "apps": APPS})
    assert config.models_for("wotan") == {"text": "mock-strong", "image": "mock-image"}
    assert config.models_for("sagadeck") == {"text": "mock-text", "image": "mock-image-pro"}
    assert config.models_for("unknown-app") == MODELS
    assert config.models_for(None) == MODELS


@pytest.mark.parametrize("apps, message", [
    ({"wotan": {"base_url": "x"}}, "Only \\[apps.wotan.models\\]"),
    ({"wotan": "text"}, "must be a table"),
    ({"wotan": {"models": {"text": 3}}}, "model strings"),
])
def test_bad_app_sections_are_explained(apps, message):
    with pytest.raises(ConfigError, match=message):
        Config.from_dict({"models": MODELS, "apps": apps})


def test_relay_resolves_with_its_app(mock):
    _, url = mock
    wotan = make_relay(url, models=MODELS, apps=APPS)
    wotan.app = "wotan"
    assert wotan.chat("hi", model="text").model == "mock-strong"
    assert wotan.chat("hi", model="image").model == "mock-image"          # not overridden -> shared
    assert wotan.chat("hi", model="text", app="sagadeck").model == "mock-text"  # per-call app wins
    assert wotan.resolve_model("unmapped-name") == "unmapped-name"        # not an alias -> passes through


@pytest.fixture
def served(mock):
    _, url = mock
    server = make_server(make_relay(url, models=MODELS, apps=APPS), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def ask(server, model, app=None, stream=False):
    headers = {APP_HEADER: app} if app else {}
    body = {"model": model, "stream": stream, "messages": [{"role": "user", "content": "oi"}]}
    return httpx.post(f"{server.url}/chat/completions", json=body, headers=headers, timeout=10)


def test_serve_uses_the_app_header(served):
    assert ask(served, "text", "wotan").json()["model"] == "mock-strong"
    assert ask(served, "image", "sagadeck").json()["model"] == "mock-image-pro"
    assert ask(served, "text").json()["model"] == "mock-text"               # no header -> [models]
    assert ask(served, "text", "someone-else").json()["model"] == "mock-text"


def test_stream_resolves_with_the_app_too(mock, served):
    _, url = mock
    relay = make_relay(url, models=MODELS, apps=APPS)
    done = [ev for ev in relay.stream("hi", model="text", app="wotan") if ev.type == "done"][0]
    assert done.response.model == "mock-strong"                       # what the provider received
    r = ask(served, "text", "wotan", stream=True)                       # and through serve, with the header
    assert r.status_code == 200 and r.text.rstrip().endswith("data: [DONE]")


def test_models_endpoint_lists_what_the_app_sees(served):
    ids = lambda app=None: [m["id"] for m in httpx.get(f"{served.url}/models", headers={APP_HEADER: app} if app else {}).json()["data"]]
    assert ids() == ["text", "image"]
    assert ids("wotan") == ["text", "image"]


def test_show_app_prints_the_merged_models(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[models]\n"text" = "a"\n"image" = "b"\n[apps.wotan.models]\n"text" = "strong"\n', encoding="utf-8")
    monkeypatch.setenv("MODELRELAY_CONFIG", str(cfg))
    assert main(["show", "--app", "wotan"]) == 0
    out = capsys.readouterr().out
    assert "text = strong   <- [apps.wotan.models]" in out
    assert "image = b" in out
    assert main(["show", "--app", "nobody"]) == 0
    assert "no [apps.nobody] section" in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_init_template_documents_per_app_models(name):
    text = TEMPLATES[name]
    assert "[apps.wotan.models]" in text and "X-Modelrelay-App" in text
    Config.from_dict(tomllib.loads(text))  # still a valid config (the example is commented out)
