# modelrelay

One small API to call LLMs anywhere. Write your code once, against `modelrelay`,
and decide **by configuration** where calls go:

- any **OpenAI-compatible** API: OpenAI, OpenRouter, an internal LLM proxy;
- a **job-queue gateway** (submit a request, then poll until it's done);
- with a fixed **API key**, or **client credentials** that turn into a token that expires.

Anything specific to one company's gateway lives in a small private adapter
package that plugs in. It never goes into your projects or into this library.

```python
from modelrelay import llm

resp = llm.chat("Explain recursion in one sentence", model="gpt-4o")
print(resp.text)
```

## Install

```bash
pip install git+https://github.com/naruminho/modelrelay
modelrelay init      # once per machine: creates ~/.modelrelay/config.toml (OpenRouter template)
modelrelay show      # prints the config in use, secrets masked
```

`pip install` can't create files in your home folder, so `modelrelay init` does it. Other
templates: `modelrelay init --template openai` or `--template gateway`. `init` never overwrites
an existing file.

## Usage

```python
from modelrelay import Relay

llm = Relay()  # reads the config (see below)

# You own the history: send whatever messages you want each time.
history = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Hi!"},
]
resp = llm.chat(history, model="gpt-4o", temperature=0.2)
history.append(resp.to_message())

# Images / PDFs go to the last user message
llm.chat("What's in these?", model="gpt-4o", files=["photo.png", "invoice.pdf"])

# Image generation through multimodal models
resp = llm.chat("A cat writing code", model="some-image-model")
resp.images[0].save("cat.png")
```

### Streaming and status events

`stream()` yields events. The last one is always `done`, with the full response.

```python
for ev in llm.stream(history, model="gpt-4o"):
    if ev.type == "delta":       # text as it's generated (when the transport can stream)
        print(ev.text, end="")
    elif ev.type in ("queued", "running"):   # job-based transports
        print(f"\rthinking... {ev.elapsed:.0f}s", end="")
    elif ev.type == "done":
        final = ev.response
```

The same code works everywhere. With a job gateway you get status updates instead of
text deltas; `llm.supports_text_stream` tells you which one to expect.

### Tool calling

```python
tools = [{
    "name": "run_command",
    "description": "Runs a shell command",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
}]

resp = llm.chat(history, model="gpt-4o", tools=tools)
for call in resp.tool_calls:
    output = my_tools[call.name](**call.arguments)
    history += [resp.to_message(), {"role": "tool", "tool_call_id": call.id, "content": output}]
```

With `tools_mode = "emulated"`, tools are described in the prompt and the answer is parsed
back into `resp.tool_calls`. Use it for providers without native tool calling. Your code
doesn't change.

## Errors and tracing

modelrelay never hides a failure. It does not guess missing fields, turn unknown statuses
into "running", or hand back half an answer as if it were complete. Every problem is an
exception that inherits from `ModelRelayError`:

| Exception | When |
|---|---|
| `ProviderError` | HTTP error, network error, or an `error` field in the response |
| `UnexpectedResponse` | the response doesn't have the expected shape (the contract changed?); never retried |
| `StreamInterrupted` | the stream ended before the provider finished; `.partial` holds what arrived |
| `JobFailed` / `JobTimeout` | the job reported an error, or did not finish in `max_wait_seconds` |
| `InvalidToolCall` | the model called a tool with arguments that aren't a JSON object; `.raw` and `.response` included |
| `AuthError` | the token could not be obtained |
| `PayloadTooLarge` | the request is above `max_payload_mb`; checked before sending |
| `ConfigError` | invalid or missing configuration |

Every error carries its context. It is printed with the message and available in `error.context`:

```
HTTP 504: upstream request timeout [status=504, method=POST, url=https://.../chat/completions,
trace_id=3f9c1a2b7d4e, elapsed=30.02, transport=OpenAICompatible, model=region1;gpt-4o]
```

`error.body` has the full, untruncated response. The original exception stays in `__cause__`.

**Trace id.** Each call gets a `trace_id`. It appears in logs, errors, events and on
`response.trace_id`.

**Logs.** Set `MODELRELAY_LOG=debug` (or call `modelrelay.enable_logging()`) to see every HTTP
call with its status and timing, token renewals, job status changes and each polling retry.
Tokens and message contents are never logged.

## What adapters can change

Adapters can change anything specific to a gateway, without touching modelrelay:

- **URLs, request body and response parsing:** override the hooks in `JobsTransport` /
  `OpenAICompatible`, or subclass `Transport` and write `complete()` and `stream()` from scratch.
  See [Private adapters](#private-adapters) and [ADAPTER_GUIDE.md](ADAPTER_GUIDE.md).
  Relay needs nothing else.
- **Headers and fixed body fields:** `extra_headers` / `extra_body` in the config, or override
  `headers_for()` for per-request headers.
- **Status names:** map them in `parse_poll()`, and raise on anything unknown.
- **Authentication:** `client_credentials` options, or a `TokenProvider` of your own.
- **Anything the core doesn't model:** `response.raw` keeps the original payload, and
  `**params` in `chat()` goes straight to the transport.

The public API (`chat`, `stream`, `Response`, `Event`, the errors) and the adapter hooks
follow semantic versioning. A breaking change to either means a new major version.

## Configuration

Configs live in one fixed folder, `~/.modelrelay/` (`C:\Users\<you>\.modelrelay\` on Windows).
No environment variable is needed. `config.toml` is the default. One file is usually all you
need, since it can send `chat()` and `stream()` through different paths (see the gateway example).
Extra files are optional **profiles**, for when you want to switch whole setups:

```python
llm = Relay()                     # ~/.modelrelay/config.toml
llm = Relay(profile="openai")     # ~/.modelrelay/openai.toml  (modelrelay init --profile openai --template openai)
llm = Relay(config_path="somewhere/else.toml")
```

The lookup order is: `config_path`, then `profile`, then `$MODELRELAY_CONFIG` (optional, handy for
CI), then `~/.modelrelay/config.toml`. If no file is found, you get a `ConfigError` telling you to
run `modelrelay init`; there are no hidden defaults. `llm.config.source` tells you which file was
used. To build a config in code instead, use `Relay(Config.from_dict({...}))`.

**OpenRouter at home**

```toml
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[models]   # names used in code -> names the provider expects
"gpt-4o" = "openai/gpt-4o"
"gemini-2.5-flash" = "google/gemini-2.5-flash"
```

**A gateway with a job API, an OpenAI-compatible proxy and expiring tokens**
(`modelrelay init --template gateway`)

```toml
base_url = "https://gateway.example.com/v1"
transport = "gateway:GatewayJobs"      # chat(): the job API, adapter in ~/.modelrelay/adapters/gateway.py
stream_transport = "openai_compatible" # stream(): the proxy; switch to "gateway:GatewayJobs" if it goes away
tools_mode = "native"                  # or "emulated"
max_payload_mb = 20
ca_bundle = "C:/certs/company-ca.pem"  # or verify_ssl = false (not recommended)

auth = "client_credentials"
[auth_options]
token_url = "https://identity.example.com/token"
token_field = "data.token"   # dotted path in the response
ttl_minutes = 30             # renewed 2 minutes before it expires
client_id = "..."            # or leave them out and set $MODELRELAY_CLIENT_ID /
client_secret = "..."        # $MODELRELAY_CLIENT_SECRET instead

[models]
"gpt-4o" = "region1;gpt-4o"

[transports."gateway:GatewayJobs"]   # each path can have its own base_url
base_url = "https://gateway.example.com/jobs-api"
poll_interval = 0.5
poll_max_interval = 5
max_wait_seconds = 900

[transports.openai_compatible]
base_url = "https://proxy.example.com/v1"
```

Both paths share the same token.

Every option:

| Key | Default | Meaning |
|---|---|---|
| `transport` | `openai_compatible` | transport for `chat()` (builtin: `openai_compatible`, `jobs`; or a plugin) |
| `stream_transport` | same as `transport` | transport for `stream()` |
| `base_url` | OpenAI | API root |
| `auth` | `static` | `static`, `client_credentials` or a plugin |
| `api_key` / `api_key_env` | – / `OPENAI_API_KEY` | key for `static` |
| `auth_options` | `{}` | `token_url`, `token_field`, `ttl_minutes`, `refresh_margin_seconds`, `request_format` (`json`/`form`), `id_field`, `secret_field`, `extra_fields`, `client_id`, `client_secret` (or `client_id_env` / `client_secret_env` to read them from other env vars) |
| `models` | `{}` | model name map |
| `tools_mode` | `native` | `native` or `emulated` (can also be set per transport) |
| `verify_ssl` / `ca_bundle` | `true` / – | TLS verification |
| `timeout_seconds` | `120` | HTTP timeout |
| `max_payload_mb` | – | refuse requests bigger than this before sending |
| `headers` | `{}` | extra headers on every request |
| `transports.<name>` | `{}` | options for one transport: `base_url`, `tools_mode`, `extra_body`, `extra_headers`, polling settings |

Tokens are sent as `Authorization: Bearer <token>`. When a request gets 401/403, a
`client_credentials` token is renewed once and the request retried.

## Private adapters

When a gateway doesn't match the defaults, an adapter translates its contract. An adapter is
**one Python file on your machine**, next to the config:

```
~/.modelrelay/                     (C:\Users\<you>\.modelrelay\ on Windows)
├── config.toml                    transport = "gateway:GatewayJobs"
└── adapters/
    └── gateway.py                 class GatewayJobs(JobsTransport)
```

`modelrelay init --template gateway` creates both, with the adapter as a skeleton full of TODOs.
**[ADAPTER_GUIDE.md](ADAPTER_GUIDE.md)** walks through filling it in, step by step. It is written so
an AI agent can follow it. A filled-in adapter looks like this:

```python
# ~/.modelrelay/adapters/gateway.py
from modelrelay import JobsTransport, JobState, UnexpectedResponse, require
from modelrelay.transports import parse_completion

class GatewayJobs(JobsTransport):
    def submit_url(self, req): return f"{self.base_url}/start"
    def poll_url(self, job_id, req): return f"{self.base_url}/status/{job_id}"
    def build_submit(self, req): return {"flow": "llm", "inputs": {"model": req.model, "messages": req.messages}}
    def parse_submit(self, data): return require(data, "execution.id")   # raises if missing
    def parse_poll(self, data):
        state = require(data, "execution.state")
        if state == "DONE":
            return JobState("done", response=parse_completion(require(data, "output")), raw=data)
        if state == "ERROR":
            return JobState("error", error=require(data, "output.message"), raw=data)
        if state in ("QUEUED", "RUNNING"):
            return JobState(state.lower(), raw=data)
        raise UnexpectedResponse(f"Unknown state {state!r}", body=data)   # never guess
```

modelrelay does the polling, retries, timeouts, token renewal and error reporting. The adapter
only says where to send, what to send and how to read the answers.
[`mock_adapter.py`](src/modelrelay/testing/mock_adapter.py) is a complete working example.

In `transport` and `auth`, `"file:Class"` loads `Class` from `~/.modelrelay/adapters/file.py`.
Your projects stay clean: they only `pip install` modelrelay and call `llm.chat(...)`. At home the
same projects run with a different `config.toml` (for example OpenRouter) and no adapter at all.

Keep the adapter out of public repositories. If you want it versioned, put `~/.modelrelay/adapters/`
in your company's git, never the config (it holds credentials).

**Alternative: an installable package.** If you'd rather ship the adapter as a package, register it
with entry points and name it in the config:

```toml
# pyproject.toml of your private package
[project.entry-points."modelrelay.transports"]
my_gateway_jobs = "my_gateway_adapter.transport:GatewayJobs"

[project.entry-points."modelrelay.auth"]             # only if the token exchange is unusual
my_identity = "my_gateway_adapter.auth:MyIdentity"   # subclass ClientCredentials, override fetch_token()
```

Then `transport = "my_gateway_jobs"`, and every project installs the package too.

## Mock gateway

A fake gateway to develop against without spending tokens: expiring tokens, a request time
limit, a job queue with nested payloads, tool calls and image output.

```bash
python -m modelrelay.testing.mock_server --port 8000 --token-ttl 120
```

```toml
base_url = "http://127.0.0.1:8000/v1"
transport = "modelrelay.testing.mock_adapter:MockJobsTransport"
auth = "client_credentials"
[auth_options]
token_url = "http://127.0.0.1:8000/auth/token"
token_field = "data.token"
```

Any client id/secret is accepted; a static key `test-key` also works.

## Development

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"   # or .venv/bin/pip
.venv/Scripts/pytest
```

`pytest` runs against the mock and costs nothing. To check a real provider end to end
(chat, history, stream, native/emulated tools, image and PDF input, image generation, errors):

```bash
python examples/smoke_test.py                       # OpenRouter, needs $OPENROUTER_API_KEY
python examples/smoke_test.py my-config.toml --text-model X --vision-model Y --image-model Z
```

## License

MIT
