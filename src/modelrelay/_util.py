from __future__ import annotations

import logging
import os
from importlib import import_module
from importlib.metadata import entry_points

from .errors import ConfigError, UnexpectedResponse

log = logging.getLogger("modelrelay")
_MISSING = object()


def dig(obj, path: str, default=None):
    """dig({"a": {"b": [1, 2]}}, "a.b.1") -> 2. Returns `default` when the path is missing."""
    for key in path.split("."):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        elif isinstance(obj, list) and key.isdigit() and int(key) < len(obj):
            obj = obj[int(key)]
        else:
            return default
    return obj


def require(obj, path: str, what: str = ""):
    """Like dig(), but raises UnexpectedResponse (with the full payload) when the path is missing."""
    value = dig(obj, path, _MISSING)
    if value is _MISSING or value is None:
        label = f" ({what})" if what else ""
        raise UnexpectedResponse(f"Field '{path}'{label} not found in response", body=obj)
    return value


def load_object(name: str, group: str, builtins: dict):
    """Resolve a builtin name, a 'package.module:Class' path, or an installed entry point."""
    if name in builtins:
        return builtins[name]
    if ":" in name:
        module, attr = name.split(":", 1)
        return getattr(import_module(module), attr)
    for ep in entry_points(group=group):
        if ep.name == name:
            return ep.load()
    raise ConfigError(
        f"Unknown '{name}' for {group}. Use one of {sorted(builtins)}, a 'module:Class' path, "
        f"or install a package that registers it under the '{group}' entry point group."
    )


def enable_logging(level: str | int = "DEBUG") -> None:
    """Prints modelrelay's logs to stderr. Same as setting MODELRELAY_LOG=debug."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(level.upper() if isinstance(level, str) else level)


if os.environ.get("MODELRELAY_LOG"):
    enable_logging(os.environ["MODELRELAY_LOG"])
