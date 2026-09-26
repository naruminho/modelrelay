"""Command line: `modelrelay init` creates a config, `modelrelay show` prints the one in use,
`modelrelay serve` exposes it as a local OpenAI-compatible endpoint, with a setup screen at /."""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files

from ._util import adapters_dir
from .config import DEFAULT_PROFILE, Config, config_dir
from .errors import ModelRelayError

TEMPLATES = {
    "openrouter": '''\
# modelrelay config: OpenRouter
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"   # or put the key here: api_key = "sk-or-..."

# Roles: apps ask for a role, this file picks the model.
#   "text"  reads: conversation, and the images the app sends (pick a model with image input if the app sends images)
#   "image" generates images (a model with image output)
# A role can carry default params, e.g. how hard the model thinks:
#   "text" = { model = "provider/model", reasoning_effort = "medium" }   # minimal | low | medium | high
[models]   # name used in code = name the provider expects
"text" = "google/gemini-2.5-flash"               # reads (text + images)
"image" = "google/gemini-2.5-flash-image"        # generates images
"gpt-4o-mini" = "openai/gpt-4o-mini"
"gemini-2.5-flash" = "google/gemini-2.5-flash"
"gemini-2.5-flash-image" = "google/gemini-2.5-flash-image"

# Per-app models (optional). Provider and credentials above are shared by every app; an app
# section only lists the names it wants resolved differently. Apps say who they are with the
# X-Modelrelay-App header (modelrelay serve) or Relay(app="..."). Check with: modelrelay show --app <app>
# [apps.wotan.models]
# "text" = { model = "anthropic/claude-opus-5-5", reasoning_effort = "high" }
# [apps.sagadeck.models]
# "image" = "google/gemini-3-pro-image"
''',
    "openai": '''\
# modelrelay config: OpenAI
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"   # or put the key here: api_key = "sk-..."

# Roles: apps ask for a role, this file picks the model.
#   "text"  reads: conversation, and the images the app sends (pick a model with image input if the app sends images)
#   "image" generates images (a model with image output)
# A role can carry default params, e.g. how hard the model thinks:
#   "text" = { model = "provider/model", reasoning_effort = "medium" }   # minimal | low | medium | high
[models]
"text" = "gpt-4o"            # reads (text + images)

# Per-app models (optional). Provider and credentials above are shared by every app; an app
# section only lists the names it wants resolved differently. Apps say who they are with the
# X-Modelrelay-App header (modelrelay serve) or Relay(app="..."). Check with: modelrelay show --app <app>
# [apps.wotan.models]
# "text" = { model = "anthropic/claude-opus-5-5", reasoning_effort = "high" }
# [apps.sagadeck.models]
# "image" = "google/gemini-3-pro-image"
''',
    "gateway": '''\
# modelrelay config: a gateway with a job API, an OpenAI-compatible proxy and expiring tokens.
# One file uses both paths with the same token. Fill in the values, then run: modelrelay show
# The job API adapter is adapters/gateway.py, next to this file.
base_url = "https://gateway.example.com/v1"

transport = "gateway:GatewayJobs"        # chat(): the job API (adapters/gateway.py, class GatewayJobs)
stream_transport = "openai_compatible"   # stream(): the proxy, text as it is generated
                                         # if the proxy goes away: stream_transport = "gateway:GatewayJobs"
tools_mode = "native"                    # or "emulated"
max_payload_mb = 20
# ca_bundle = "C:/certs/company-ca.pem"  # or: verify_ssl = false

auth = "client_credentials"
[auth_options]
token_url = "https://identity.example.com/token"
token_field = "access_token"             # dotted path in the token response
ttl_minutes = 30
client_id = ""                           # this file stays in your user folder; `modelrelay show` masks these
client_secret = ""

# Roles: apps ask for a role, this file picks the model.
#   "text"  reads: conversation, and the images the app sends (pick a model with image input if the app sends images)
#   "image" generates images (a model with image output)
# A role can carry default params, e.g. how hard the model thinks:
#   "text" = { model = "provider/model", reasoning_effort = "medium" }   # minimal | low | medium | high
[models]
# "gpt-4o" = "region;gpt-4o"
# "text" = "region;gpt-4o"   # reads (text + images)

# Per-app models (optional). Provider and credentials above are shared by every app; an app
# section only lists the names it wants resolved differently. Apps say who they are with the
# X-Modelrelay-App header (modelrelay serve) or Relay(app="..."). Check with: modelrelay show --app <app>
# [apps.wotan.models]
# "text" = { model = "anthropic/claude-opus-5-5", reasoning_effort = "high" }
# [apps.sagadeck.models]
# "image" = "google/gemini-3-pro-image"

[transports."gateway:GatewayJobs"]
base_url = "https://gateway.example.com/jobs-api"
poll_interval = 0.5
poll_max_interval = 5
max_wait_seconds = 900

[transports.openai_compatible]
base_url = "https://proxy.example.com/v1"
''',
}

# Extra files created next to a template's config (path relative to ~/.modelrelay).
TEMPLATE_FILES = {"gateway": {"adapters/gateway.py": "gateway_adapter.py"}}

SECRET_KEYS = {"api_key", "client_secret", "client_id"}


def init(profile: str, template: str) -> int:
    path = config_dir() / f"{profile}.toml"
    if path.exists():
        print(f"{path} already exists; nothing changed. Edit it, or delete it and run init again.")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATES[template], encoding="utf-8")
    print(f"Created {path} from the '{template}' template.")
    for relative, resource in TEMPLATE_FILES.get(template, {}).items():
        extra = config_dir() / relative
        if extra.exists():
            print(f"Kept the existing {extra}.")
            continue
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text((files("modelrelay") / "templates" / resource).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created {extra}.")
    print("Edit the file(s), then check with: modelrelay show"
          + (f" --profile {profile}" if profile != DEFAULT_PROFILE else ""))
    return 0


def show(profile: str | None, app: str | None = None) -> int:
    config = Config.load(profile=profile)
    print(f"config file: {config.source}")
    print(f"adapters folder: {adapters_dir()}")
    if app:
        own = (config.apps.get(app) or {}).get("models", {})
        print(f"models for app '{app}'" + ("" if app in config.apps else f" (no [apps.{app}] section: using [models])") + ":")
        for name in config.entries_for(app):
            target, params = config.route(name, app)
            provider = config.provider_for(name, app)
            params = {**({"provider": provider} if provider and config.providers else {}), **params}
            extra = "  (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")" if params else ""
            print(f"  {name} = {target}{extra}" + ("   <- [apps.%s.models]" % app if name in own else ""))
        return 0
    for name, value in vars(config).items():
        if name == "source":
            continue
        print(f"{name} = {_mask(name, value)}")
    return 0


def _mask(name, value):
    if isinstance(value, dict):
        return {k: _mask(k, v) for k, v in value.items()}
    if name in SECRET_KEYS and value:
        return "***"
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelrelay")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help=f"create {config_dir()}/<profile>.toml")
    p_init.add_argument("--profile", default=DEFAULT_PROFILE)
    p_init.add_argument("--template", choices=sorted(TEMPLATES), default="openrouter")

    p_show = sub.add_parser("show", help="print the config in use (secrets masked)")
    p_show.add_argument("--profile")
    p_show.add_argument("--app", help="only the models this app gets ([models] + [apps.<app>.models])")

    p_serve = sub.add_parser("serve", help="local OpenAI-compatible endpoint (/v1/chat/completions) using the config")
    p_serve.add_argument("--profile")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--api-key", help="require this bearer token (default: $MODELRELAY_SERVE_KEY, else none)")
    p_serve.add_argument("--public-url", help="address of the setup screen behind a login proxy, e.g. https://example.com/ia/")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return init(args.profile, args.template)
        if args.command == "serve":
            from .server import serve
            serve(host=args.host, port=args.port, api_key=args.api_key, profile=args.profile, public_url=args.public_url)
            return 0
        return show(args.profile, args.app)
    except ModelRelayError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
