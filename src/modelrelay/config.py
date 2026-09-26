from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from .errors import ConfigError

CONFIG_ENV = "MODELRELAY_CONFIG"  # optional override, e.g. for CI
DEFAULT_PROFILE = "config"


def config_dir() -> Path:
    """~/.modelrelay, i.e. C:/Users/<user>/.modelrelay on Windows."""
    return Path.home() / ".modelrelay"


@dataclass
class Config:
    # Which transport handles chat() and stream(). stream_transport falls back to transport.
    transport: str = "openai_compatible"
    stream_transport: str | None = None
    base_url: str = "https://api.openai.com/v1"

    # "static" (fixed API key), "client_credentials" (id + secret -> expiring token), or a plugin.
    auth: str = "static"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    auth_options: dict = field(default_factory=dict)

    # Model names used in code -> names the provider expects. A value can also be a table with
    # default request params: "text" = { model = "provider/model", reasoning_effort = "high" }
    models: dict = field(default_factory=dict)

    # Per-app model overrides: [apps.<app>.models]. An app only lists what differs from [models];
    # provider, auth and everything else stay shared. See models_for().
    apps: dict = field(default_factory=dict)

    # "native" sends `tools` to the API; "emulated" describes them in the prompt and parses the reply.
    tools_mode: str = "native"

    verify_ssl: bool = True
    ca_bundle: str | None = None
    timeout_seconds: float = 120.0
    max_payload_mb: float | None = None
    headers: dict = field(default_factory=dict)

    # Per-transport options, e.g. [transports.jobs] poll_interval = 1.0
    transports: dict = field(default_factory=dict)

    # More than one provider: [providers.<name>] holds the same connection keys as the top of the
    # file (base_url, auth, api_key, transport...). A model entry picks one with `provider = "<name>"`;
    # entries without it use `provider` below, or the connection at the top of the file when unset.
    providers: dict = field(default_factory=dict)
    provider: str | None = None

    @classmethod
    def load(cls, path: str | Path | None = None, profile: str | None = None, **overrides) -> Config:
        """Loads the config file, in this order:

        1. `path`, if given
        2. ~/.modelrelay/<profile>.toml, if `profile` is given (must exist)
        3. $MODELRELAY_CONFIG, if set (optional; never required)
        4. ~/.modelrelay/config.toml

        No file found is an error (run `modelrelay init`). To build a config in code
        instead, use Config.from_dict(). The file used is in `config.source`.
        """
        source = _locate(path, profile)
        data = _read(source)
        data.update(overrides)
        config = cls.from_dict(data)
        config.source = source
        return config

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        known = {f.name for f in fields(cls) if f.init}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
        config = cls(**data)
        if config.tools_mode not in ("native", "emulated"):
            raise ConfigError("tools_mode must be 'native' or 'emulated'")
        _check_models(config.models, "[models]")
        _check_apps(config.apps)
        _check_providers(config)
        return config

    def provider_for(self, name: str, app: str | None = None) -> str | None:
        """Which [providers.<name>] serves a model name (None: the connection at the top of the file)."""
        entry = self.entries_for(app).get(name)
        own = entry.get("provider") if isinstance(entry, dict) else None
        return own or self.provider

    def for_provider(self, name: str) -> Config:
        """The connection settings of one provider, as a standalone Config. Nothing is inherited from
        the top of the file, so one provider's key never reaches another provider's URL."""
        config = Config.from_dict(dict(self.providers[name]))
        config.source = self.source
        return config

    def entries_for(self, app: str | None = None) -> dict:
        """Raw entries an app sees: [models], with [apps.<app>.models] on top (a whole entry replaces
        the shared one). Unknown or missing app -> just [models]."""
        own = (self.apps.get(app) or {}).get("models", {}) if app else {}
        return {**self.models, **own}

    def models_for(self, app: str | None = None) -> dict:
        """Name -> provider model for an app (params left out; see route())."""
        return {name: _split(entry)[0] for name, entry in self.entries_for(app).items()}

    def route(self, name: str, app: str | None = None) -> tuple[str, dict]:
        """(provider model, default request params) for a name. Not an alias -> (name, {})."""
        entry = self.entries_for(app).get(name)
        return _split(entry) if entry is not None else (name, {})

    source: Path | None = field(default=None, init=False, repr=False, compare=False)

    def transport_options(self, name: str) -> dict:
        return dict(self.transports.get(name, {}))


def _split(entry) -> tuple[str, dict]:
    if isinstance(entry, str):
        return entry, {}
    params = {k: v for k, v in entry.items() if k not in ("model", "provider")}
    return entry["model"], params


# Keys a [providers.<name>] section can hold: the connection, not the model maps.
PROVIDER_KEYS = {"transport", "stream_transport", "base_url", "auth", "api_key", "api_key_env", "auth_options",
                 "tools_mode", "verify_ssl", "ca_bundle", "timeout_seconds", "max_payload_mb", "headers", "transports"}


def _check_providers(config) -> None:
    if not isinstance(config.providers, dict):
        raise ConfigError("providers must be tables: [providers.<name>]")
    for name, section in config.providers.items():
        if not isinstance(section, dict):
            raise ConfigError(f'[providers.{name}] must be a table, e.g. [providers.{name}] base_url = "https://..."')
        unknown = set(section) - PROVIDER_KEYS
        if unknown:
            raise ConfigError(f"Unknown keys in [providers.{name}]: {sorted(unknown)}. A provider only holds the "
                              f"connection ({', '.join(sorted(PROVIDER_KEYS))}); models go in [models].")
    used = [("provider", config.provider)] if config.provider else []
    for where, models in [("[models]", config.models)] + [
            (f"[apps.{a}.models]", (s or {}).get("models", {})) for a, s in config.apps.items()]:
        used += [(f'{where} "{n}"', e.get("provider")) for n, e in models.items() if isinstance(e, dict) and e.get("provider")]
    for where, name in used:
        if name not in config.providers:
            known = ", ".join(sorted(config.providers)) or "none yet"
            raise ConfigError(f'{where} uses provider "{name}", but there is no [providers.{name}] (known: {known})')


def _check_models(models, where: str) -> None:
    if not isinstance(models, dict):
        raise ConfigError(f"{where} must map names to models")
    for name, entry in models.items():
        if isinstance(entry, str):
            continue
        if not isinstance(entry, dict):
            raise ConfigError(f'{where} "{name}" must be a model string or a table, e.g. '
                              f'"{name}" = {{ model = "provider/model", reasoning_effort = "high" }}')
        if not isinstance(entry.get("model"), str) or not entry["model"]:
            raise ConfigError(f'{where} "{name}" needs a `model`: '
                              f'"{name}" = {{ model = "provider/model", reasoning_effort = "high" }}')


def _check_apps(apps) -> None:
    if not isinstance(apps, dict):
        raise ConfigError("apps must be a table: [apps.<app>.models]")
    for name, section in apps.items():
        if not isinstance(section, dict):
            raise ConfigError(f"[apps.{name}] must be a table with a [apps.{name}.models] section")
        unknown = set(section) - {"models"}
        if unknown:
            raise ConfigError(f"Unknown keys in [apps.{name}]: {sorted(unknown)}. Only [apps.{name}.models] is supported: "
                              "provider, auth and transports are shared by all apps (use a profile to change those).")
        _check_models(section.get("models", {}), f"[apps.{name}.models]")


def _locate(path, profile) -> Path:
    if path:
        return _must_exist(Path(path), "Config file")
    if profile:
        path = config_dir() / f"{profile}.toml"
        if not path.is_file():
            raise ConfigError(f"Profile '{profile}' not found: {path}. "
                              f"Run `modelrelay init --profile {profile}` to create it.")
        return path
    if os.environ.get(CONFIG_ENV):
        return _must_exist(Path(os.environ[CONFIG_ENV]), f"${CONFIG_ENV}")
    default = config_dir() / f"{DEFAULT_PROFILE}.toml"
    if not default.is_file():
        raise ConfigError(f"No config found at {default}. Run `modelrelay init` to create one.")
    return default


def _must_exist(path: Path, what: str) -> Path:
    if not path.is_file():
        raise ConfigError(f"{what} not found: {path}")
    return path


def _read(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e
