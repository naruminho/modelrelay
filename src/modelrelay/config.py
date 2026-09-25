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

    # Model names used in code -> names the provider expects.
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
        _check_apps(config.apps)
        return config

    def models_for(self, app: str | None = None) -> dict:
        """The model map an app sees: [models], with [apps.<app>.models] on top.
        Unknown or missing app -> just [models]."""
        own = (self.apps.get(app) or {}).get("models", {}) if app else {}
        return {**self.models, **own}

    source: Path | None = field(default=None, init=False, repr=False, compare=False)

    def transport_options(self, name: str) -> dict:
        return dict(self.transports.get(name, {}))


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
        models = section.get("models", {})
        if not isinstance(models, dict) or not all(isinstance(v, str) for v in models.values()):
            raise ConfigError(f'[apps.{name}.models] must map names to model strings, e.g. "text" = "provider/model"')


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
