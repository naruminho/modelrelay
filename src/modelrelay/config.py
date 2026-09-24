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

CONFIG_ENV = "MODELRELAY_CONFIG"
DEFAULT_LOCATIONS = (Path("modelrelay.toml"), Path.home() / ".modelrelay.toml")


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
    def load(cls, path: str | Path | None = None, **overrides) -> Config:
        """Load from `path`, $MODELRELAY_CONFIG, ./modelrelay.toml or ~/.modelrelay.toml (first found)."""
        data: dict = {}
        path = path or os.environ.get(CONFIG_ENV)
        if path:
            if not Path(path).is_file():
                raise ConfigError(f"Config file not found: {path}")
            data = _read(Path(path))
        else:
            for candidate in DEFAULT_LOCATIONS:
                if candidate.is_file():
                    data = _read(candidate)
                    break
        data.update(overrides)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
        config = cls(**data)
        if config.tools_mode not in ("native", "emulated"):
            raise ConfigError("tools_mode must be 'native' or 'emulated'")
        return config

    def transport_options(self, name: str) -> dict:
        return dict(self.transports.get(name, {}))


def _read(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)
