"""Command line: `modelrelay init` creates a config, `modelrelay show` prints the one in use,
`modelrelay serve` exposes it as a local OpenAI-compatible endpoint."""

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

[models]   # name used in code = name the provider expects
"text" = "google/gemini-2.5-flash"               # role aliases: apps ask for "text"/"image",
"image" = "google/gemini-2.5-flash-image"        # the config decides the real model
"gpt-4o-mini" = "openai/gpt-4o-mini"
"gemini-2.5-flash" = "google/gemini-2.5-flash"
"gemini-2.5-flash-image" = "google/gemini-2.5-flash-image"
''',
    "openai": '''\
# modelrelay config: OpenAI
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"   # or put the key here: api_key = "sk-..."

[models]
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

[models]
# "gpt-4o" = "region;gpt-4o"

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


def show(profile: str | None) -> int:
    config = Config.load(profile=profile)
    print(f"config file: {config.source}")
    print(f"adapters folder: {adapters_dir()}")
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

    p_serve = sub.add_parser("serve", help="local OpenAI-compatible endpoint (/v1/chat/completions) using the config")
    p_serve.add_argument("--profile")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--api-key", help="require this bearer token (default: $MODELRELAY_SERVE_KEY, else none)")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return init(args.profile, args.template)
        if args.command == "serve":
            from .server import serve
            serve(host=args.host, port=args.port, api_key=args.api_key, profile=args.profile)
            return 0
        return show(args.profile)
    except ModelRelayError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
