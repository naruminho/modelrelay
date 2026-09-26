"""The setup screen behind `modelrelay serve` (http://127.0.0.1:8765/): providers, keys, models, apps.

The screen edits the same config file as `modelrelay init`/`show`. This module is the part that
doesn't need a browser:

    doc = view(raw)            # config dict -> what the screen shows (secrets masked)
    raw = apply(doc, old_raw)  # what the screen sends back -> config dict (masked secrets kept)
    text = to_toml(raw)        # config dict -> file contents

Secrets never go to the browser: a saved key comes back as {"$secret": "…a3f9", "path": [...]},
and the same placeholder sent back means "keep what is in the file".
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from .config import PROVIDER_KEYS, Config
from .errors import ConfigError

SECRETS = {("api_key",), ("auth_options", "client_secret"), ("auth_options", "client_id")}

_HOSTS = [("openrouter.ai", "openrouter"), ("api.openai.com", "openai"), ("generativelanguage.googleapis.com", "google"),
          ("api.deepseek.com", "deepseek"), ("api.anthropic.com", "anthropic")]


def normalize(raw: dict) -> dict:
    """Old single-provider files become one [providers.<name>] (+ `provider`), model strings become
    tables. Same meaning, one shape for the screen."""
    raw = copy.deepcopy(raw)
    top = {k: raw.pop(k) for k in list(raw) if k in PROVIDER_KEYS}
    providers = raw.setdefault("providers", {})
    if top:
        name = _unique(_guess_name(top), providers)
        providers[name] = top
        raw.setdefault("provider", name)
    raw["models"] = {n: _entry(e) for n, e in raw.get("models", {}).items()}
    raw["apps"] = {a: {"models": {n: _entry(e) for n, e in (s or {}).get("models", {}).items()}}
                   for a, s in raw.get("apps", {}).items()}
    return raw


def view(raw: dict) -> dict:
    doc = normalize(raw)
    for name, section in doc["providers"].items():
        for path in SECRETS:
            value = _get(section, path)
            if value:
                _set(section, path, {"$secret": "…" + value[-4:] if len(value) > 8 else "…", "path": [name, *path]})
    return doc


def apply(doc: dict, old_raw: dict) -> dict:
    """The screen's document -> a valid config dict. Raises ConfigError with a readable message."""
    old = normalize(old_raw)
    raw = copy.deepcopy(doc)
    for name, section in raw.get("providers", {}).items():
        if not isinstance(section, dict):
            continue
        for path in SECRETS:
            value = _get(section, path)
            if isinstance(value, dict) and "$secret" in value:
                kept = _get(old["providers"].get(value["path"][0], {}), tuple(value["path"][1:]))
                _set(section, path, kept)
        _prune(section)
    raw = {k: v for k, v in raw.items() if v not in (None, "", {})}
    raw["models"] = _simple(raw.get("models", {}))
    for section in raw.get("apps", {}).values():
        section["models"] = _simple(section.get("models", {}))
    Config.from_dict(copy.deepcopy(raw))  # same checks as loading the file
    return raw


def save(path: Path, raw: dict) -> Path | None:
    """Writes the file; the previous one is kept as <name>.toml.bak. Returns the backup path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.is_file():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copyfile(path, backup)
    path.write_text(to_toml(raw), encoding="utf-8")
    return backup


def check_provider(section: dict, old_raw: dict, name: str | None = None) -> dict:
    """Tries one provider: lists its models (OpenAI-compatible) or gets a token (other auth).
    Returns {"ok", "message", "models"}; never raises."""
    from .relay import Relay

    try:
        section = apply({"providers": {name or "_": section}}, old_raw if name else {})["providers"][name or "_"]
        relay = Relay(Config.from_dict(copy.deepcopy(section)))
    except ConfigError as e:
        return {"ok": False, "message": str(e), "models": []}
    try:
        token = relay.auth.provider.get_token()
        if relay.config.transport != "openai_compatible":
            return {"ok": True, "message": "Credenciais aceitas (token obtido).", "models": []}
        r = relay.http.get(relay.config.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {token}"})
        if r.status_code >= 400:
            return {"ok": False, "message": f"O provedor respondeu HTTP {r.status_code}: {r.text[:300]}", "models": []}
        data = r.json().get("data", [])
        ids = sorted(m.get("id") for m in data if isinstance(m, dict) and m.get("id"))
        return {"ok": True, "message": f"Conectado: {len(ids)} modelos disponíveis.", "models": ids}
    except Exception as e:  # network, TLS, bad JSON... all end up on the screen
        return {"ok": False, "message": f"{type(e).__name__}: {e}", "models": []}
    finally:
        relay.close()


# ---- TOML ----------------------------------------------------------------------------------

def to_toml(raw: dict) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# modelrelay: salvo pela tela de configuração em {stamp} (o anterior fica em .bak).",
             "# Pode editar à mão também; confira com: modelrelay show", ""]
    order = ["provider", "providers", "models", "apps"]  # connection first, then what uses it
    _table(lines, [], {k: raw[k] for k in sorted(raw, key=lambda k: (order.index(k) if k in order else -1))})
    return "\n".join(lines).rstrip() + "\n"


def _table(lines, path, table):
    inline_values = path and path[-1] == "models"  # model entries stay one per line
    subtables = []
    for key, value in table.items():
        if isinstance(value, dict) and not inline_values:
            subtables.append((key, value))
        else:
            lines.append(f"{_key(key)} = {_value(value)}")
    for key, value in subtables:
        if any(not isinstance(v, dict) or (key == "models") for v in value.values()) or not value:
            lines += ["", f"[{'.'.join(_key(k) for k in path + [key])}]"]
        _table(lines, path + [key], value)


def _key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key, ensure_ascii=False)


def _value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }"
    raise ConfigError(f"Can't write {type(value).__name__} to TOML")


# ---- helpers ---------------------------------------------------------------------------------

def _simple(models: dict) -> dict:
    """{ model = "x" } alone is written as "x", like people write it by hand."""
    return {n: e["model"] if isinstance(e, dict) and set(e) == {"model"} else e for n, e in models.items()}


def _entry(entry):
    return {"model": entry} if isinstance(entry, str) else dict(entry)


def _guess_name(section: dict) -> str:
    url = section.get("base_url", "https://api.openai.com/v1")
    return next((name for host, name in _HOSTS if host in url), "principal")


def _unique(name: str, taken: dict) -> str:
    n, i = name, 2
    while n in taken:
        n, i = f"{name}{i}", i + 1
    return n


def _get(d, path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _set(d, path, value):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    if value is None:
        d.pop(path[-1], None)
    else:
        d[path[-1]] = value


def _prune(d: dict) -> None:
    """Empty fields in the form mean "not set"."""
    for k in list(d):
        if isinstance(d[k], dict):
            _prune(d[k])
        if d[k] in (None, "", {}):
            del d[k]
