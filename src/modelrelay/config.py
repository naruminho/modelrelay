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

    # "native" sends `tools` to the API; "emulated" describes them in the prompt and parses the reply.
    tools_mode: str = "native"

    verify_ssl: bool = True
    ca_bundle: str | None = None
    timeout_seconds: float = 120.0
    max_payload_mb: float | None = None
    headers: dict = field(default_factory=dict)
    # If set, every request carries this header with the call's trace id (e.g. "X-Request-ID").
    trace_header: str | None = None

    # Per-transport options, e.g. [transports.jobs] poll_interval = 1.0
    transports: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None, profile: str | None = None, **overrides) -> Config:
        """Loads the config file, in this order:

        1. `path`, if given
        2. ~/.modelrelay/<profile>.toml, if `profile` is given (must exist)
        3. $MODELRELAY_CONFIG, if set (optional; never required)
        4. ~/.modelrelay/config.toml, if it exists
        5. built-in defaults (OpenAI with $OPENAI_API_KEY)

        The file that was used is in `config.source` (None for defaults).
        """
        source = _locate(path, profile)
        data = _read(source) if source else {}
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
        return config

    source: Path | None = field(default=None, init=False, repr=False, compare=False)

    def transport_options(self, name: str) -> dict:
        return dict(self.transports.get(name, {}))


def _locate(path, profile) -> Path | None:
    if path:
        return _must_exist(Path(path), "Config file")
    if profile:
        return _must_exist(config_dir() / f"{profile}.toml", f"Profile '{profile}'")
    if os.environ.get(CONFIG_ENV):
        return _must_exist(Path(os.environ[CONFIG_ENV]), f"${CONFIG_ENV}")
    default = config_dir() / f"{DEFAULT_PROFILE}.toml"
    return default if default.is_file() else None


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
