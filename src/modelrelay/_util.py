from __future__ import annotations

import importlib.util
import logging
import os
import sys
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path

from .errors import ConfigError, UnexpectedResponse

log = logging.getLogger("modelrelay")
log.addHandler(logging.NullHandler())  # silent unless the app or MODELRELAY_LOG turns logs on; errors still raise
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


def adapters_dir() -> Path:
    """~/.modelrelay/adapters: single-file adapters that live on this machine only."""
    from .config import config_dir
    return config_dir() / "adapters"


def load_object(name: str, group: str, builtins: dict):
    """Resolves, in this order:
    - a builtin name ("openai_compatible", "jobs", ...);
    - "module:Class", where module is a file in ~/.modelrelay/adapters/ (module.py);
    - "package.module:Class" importable from the environment;
    - an entry point registered under `group` by an installed package.
    """
    if name in builtins:
        return builtins[name]
    if ":" in name:
        module, attr = name.split(":", 1)
        local = adapters_dir() / f"{module}.py"
        mod = _load_file(local, module) if "." not in module and local.is_file() else import_module(module)
        if not hasattr(mod, attr):
            raise ConfigError(f"'{attr}' not found in {getattr(mod, '__file__', module)}")
        return getattr(mod, attr)
    for ep in entry_points(group=group):
        if ep.name == name:
            return ep.load()
    raise ConfigError(
        f"Unknown '{name}' for {group}. Use one of {sorted(builtins)}, 'file:Class' for a file in "
        f"{adapters_dir()}, a 'package.module:Class' path, or an entry point in the '{group}' group."
    )


def _load_file(path: Path, module: str):
    key = f"modelrelay_adapters.{module}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[key]
        raise
    log.debug("loaded adapter %s", path)
    return mod


def enable_logging(level: str | int = "DEBUG") -> None:
    """Prints modelrelay's logs to stderr. Same as setting MODELRELAY_LOG=debug."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(level.upper() if isinstance(level, str) else level)


if os.environ.get("MODELRELAY_LOG"):
    enable_logging(os.environ["MODELRELAY_LOG"])
