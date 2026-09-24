"""Command line: `modelrelay init` creates a config, `modelrelay show` prints the one in use."""

from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_PROFILE, Config, config_dir
from .errors import ModelRelayError

TEMPLATES = {
    "openrouter": '''\
# modelrelay config: OpenRouter
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"   # or put the key here: api_key = "sk-or-..."

[models]   # name used in code = name the provider expects
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
base_url = "https://gateway.example.com/v1"

transport = "my_gateway_jobs"            # chat(): the job API (entry point name from your adapter)
stream_transport = "openai_compatible"   # stream(): the proxy, text as it is generated
                                         # if the proxy goes away: stream_transport = "my_gateway_jobs"
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

[transports.my_gateway_jobs]
base_url = "https://gateway.example.com/jobs-api"
poll_interval = 0.5
poll_max_interval = 5
max_wait_seconds = 900

[transports.openai_compatible]
base_url = "https://proxy.example.com/v1"
''',
}

SECRET_KEYS = {"api_key", "client_secret", "client_id"}


def init(profile: str, template: str) -> int:
    path = config_dir() / f"{profile}.toml"
    if path.exists():
        print(f"{path} already exists; nothing changed. Edit it, or delete it and run init again.")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATES[template], encoding="utf-8")
    print(f"Created {path} from the '{template}' template. Edit it, then check with: modelrelay show"
          + (f" --profile {profile}" if profile != DEFAULT_PROFILE else ""))
    return 0


def show(profile: str | None) -> int:
    config = Config.load(profile=profile)
    print(f"config file: {config.source}")
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

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return init(args.profile, args.template)
        return show(args.profile)
    except ModelRelayError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
