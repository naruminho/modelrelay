"""The setup screen's backend: view/apply/save of the config file, and its HTTP routes."""
import threading
import tomllib

import httpx
import pytest

from modelrelay import Config, Relay
from modelrelay.console import apply, to_toml, view
from modelrelay.server import APP_HEADER, make_server

OLD_STYLE = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "sk-or-verysecret-a3f9",
    "models": {"text": "deepseek/deepseek-v4.1-flash", "image": {"model": "google/img", "reasoning_effort": "low"}},
    "apps": {"sagadeck": {"models": {"text": "x/y"}}},
}


def test_view_never_shows_a_secret_and_turns_old_files_into_providers():
    doc = view(OLD_STYLE)
    assert "verysecret" not in repr(doc)
    assert doc["provider"] == "openrouter"
    assert doc["providers"]["openrouter"]["api_key"] == {"$secret": "…a3f9", "path": ["openrouter", "api_key"]}
    assert doc["models"]["text"] == {"model": "deepseek/deepseek-v4.1-flash"}
    assert doc["apps"]["sagadeck"]["models"]["text"] == {"model": "x/y"}


def test_apply_keeps_masked_secrets_even_after_a_rename():
    doc = view(OLD_STYLE)
    doc["providers"]["meu-router"] = doc["providers"].pop("openrouter")
    doc["provider"] = "meu-router"
    raw = apply(doc, OLD_STYLE)
    assert raw["providers"]["meu-router"]["api_key"] == "sk-or-verysecret-a3f9"


def test_apply_takes_a_new_key_and_drops_empty_fields():
    doc = view(OLD_STYLE)
    doc["providers"]["openrouter"]["api_key"] = "sk-new"
    doc["providers"]["openrouter"]["api_key_env"] = ""
    raw = apply(doc, OLD_STYLE)
    assert raw["providers"]["openrouter"] == {"base_url": "https://openrouter.ai/api/v1", "api_key": "sk-new"}


def test_apply_refuses_a_broken_config_with_a_readable_message():
    doc = view(OLD_STYLE)
    doc["models"]["image"]["provider"] = "google"   # no such provider
    with pytest.raises(Exception, match='provider "google"'):
        apply(doc, OLD_STYLE)


def test_toml_round_trip_means_the_same_config():
    doc = view(OLD_STYLE)
    doc["providers"]["corp"] = {"base_url": "https://gw/v1", "transport": "gateway:GatewayJobs", "auth": "client_credentials",
                                "auth_options": {"token_url": "https://id/token", "ttl_minutes": 30, "client_id": "id", "client_secret": "s"},
                                "transports": {"gateway:GatewayJobs": {"poll_interval": 0.5}}, "verify_ssl": False}
    doc["apps"]["wotan"] = {"models": {"text": {"provider": "corp", "model": "region;gpt"}}}
    raw = apply(doc, OLD_STYLE)
    text = to_toml(raw)
    assert tomllib.loads(text) == raw
    config = Config.from_dict(tomllib.loads(text))
    assert config.provider_for("text", "wotan") == "corp"
    assert config.route("image") == ("google/img", {"reasoning_effort": "low"})


# ---- HTTP -----------------------------------------------------------------------------------

@pytest.fixture
def console_server(tmp_path, mock):
    _, url = mock
    path = tmp_path / "config.toml"
    path.write_text(f'base_url = "{url}/v1"\napi_key = "test-key"\n\n[models]\n"text" = "mock-a"\n', encoding="utf-8")
    server = make_server(Relay(Config.load(path)), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = server.url.replace("/v1", "")
    yield server, base, path, url
    server.shutdown()
    server.server_close()


def test_page_and_config_are_served_masked(console_server):
    server, base, path, _ = console_server
    page = httpx.get(base + "/")
    assert page.status_code == 200 and "modelrelay" in page.text
    state = httpx.get(base + "/api/console/config").json()
    assert state["path"] == str(path) and state["exists"]
    assert "test-key" not in repr(state)


def test_save_writes_the_file_keeps_a_backup_and_reloads(console_server):
    server, base, path, url = console_server
    doc = httpx.get(base + "/api/console/config").json()["config"]
    doc["models"]["text"]["model"] = "mock-b"
    r = httpx.post(base + "/api/console/save", json={"config": doc})
    assert r.status_code == 200, r.text
    assert path.with_suffix(".toml.bak").is_file()
    saved = tomllib.loads(path.read_text(encoding="utf-8"))
    assert saved["providers"][doc["provider"]]["api_key"] == "test-key"   # masked key kept
    chat = httpx.post(server.url + "/chat/completions", json={"model": "text", "messages": [{"role": "user", "content": "oi"}]})
    assert chat.json()["model"] == "mock-b"   # no restart needed


def test_save_refuses_invalid_config_and_leaves_the_file(console_server):
    _, base, path, _ = console_server
    before = path.read_text(encoding="utf-8")
    r = httpx.post(base + "/api/console/save", json={"config": {"provider": "ghost", "models": {}}})
    assert r.status_code == 400 and "ghost" in r.json()["error"]["message"]
    assert path.read_text(encoding="utf-8") == before


def test_test_button_lists_the_provider_models(console_server):
    _, base, _, url = console_server
    doc = httpx.get(base + "/api/console/config").json()["config"]
    name = doc["provider"]
    ok = httpx.post(base + "/api/console/test", json={"name": name, "provider": doc["providers"][name]}, timeout=10).json()
    assert ok["ok"] and ok["models"] == ["mock-fast", "mock-image"]
    bad = httpx.post(base + "/api/console/test", json={"name": "x", "provider": {"base_url": f"{url}/v1", "api_key": "wrong"}}, timeout=10).json()
    assert not bad["ok"] and "403" in bad["message"]


def test_apps_that_called_show_up_for_setup(console_server):
    server, base, _, _ = console_server
    httpx.post(server.url + "/chat/completions", headers={APP_HEADER: "wotan"},
               json={"model": "text", "messages": [{"role": "user", "content": "oi"}]})
    assert httpx.get(base + "/api/console/config").json()["apps_seen"] == ["wotan"]


@pytest.mark.parametrize("headers, status", [
    ({"Host": "evil.example.com"}, 403),                                   # DNS rebinding / other machine
    ({"Origin": "https://evil.example.com"}, 403),                         # another site in the browser
])
def test_only_this_machine_can_use_the_screen(console_server, headers, status):
    _, base, _, _ = console_server
    assert httpx.get(base + "/api/console/config", headers=headers).status_code == status
    assert httpx.post(base + "/api/console/save", json={"config": {}}, headers=headers).status_code == status


def test_writes_must_be_json(console_server):
    _, base, path, _ = console_server
    r = httpx.post(base + "/api/console/save", content=b'{"config": {}}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 415


def test_public_url_is_accepted_behind_a_login_proxy(mock, tmp_path):
    _, url = mock
    path = tmp_path / "config.toml"
    path.write_text(f'base_url = "{url}/v1"\napi_key = "k"\n', encoding="utf-8")
    server = make_server(Relay(Config.load(path)), port=0, public_url="https://portal.example.com/ia/")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = server.url.replace("/v1", "")
        h = {"Host": "portal.example.com", "Origin": "https://portal.example.com"}
        assert httpx.get(base + "/api/console/config", headers=h).status_code == 200
    finally:
        server.shutdown()
        server.server_close()


def test_serve_starts_without_a_config_so_the_screen_can_create_it(tmp_path, monkeypatch, mock):
    _, url = mock
    path = tmp_path / "novo" / "config.toml"
    monkeypatch.setenv("MODELRELAY_CONFIG", str(path))
    server = make_server(port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = server.url.replace("/v1", "")
        state = httpx.get(base + "/api/console/config").json()
        assert state["exists"] is False and state["path"] == str(path)
        doc = {"provider": "mock", "providers": {"mock": {"base_url": f"{url}/v1", "api_key": "test-key"}},
               "models": {"text": {"model": "mock-z"}}}
        assert httpx.post(base + "/api/console/save", json={"config": doc}).status_code == 200
        chat = httpx.post(server.url + "/chat/completions", json={"model": "text", "messages": [{"role": "user", "content": "oi"}]})
        assert chat.json()["model"] == "mock-z"
    finally:
        server.shutdown()
        server.server_close()


def test_saved_file_reads_like_a_hand_written_one():
    doc = view(OLD_STYLE)
    text = to_toml(apply(doc, OLD_STYLE))
    assert 'text = "deepseek/deepseek-v4.1-flash"' in text          # not { model = ... }
    assert text.index("[providers.") < text.index("[models]") < text.index("[apps.")


def test_show_says_which_provider_serves_each_model(tmp_path, capsys, monkeypatch):
    from modelrelay.cli import main
    path = tmp_path / "c.toml"
    path.write_text('provider = "a"\n[providers.a]\nbase_url = "http://a/v1"\n[providers.b]\nbase_url = "http://b/v1"\n'
                    '[models]\ntext = "m1"\nimage = { provider = "b", model = "m2" }\n', encoding="utf-8")
    monkeypatch.setenv("MODELRELAY_CONFIG", str(path))
    main(["show", "--app", "x"])
    out = capsys.readouterr().out
    assert "text = m1  (provider=a)" in out and "image = m2  (provider=b)" in out
