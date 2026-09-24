# Adapter guide (for AI agents and humans)

You are connecting modelrelay to a company gateway. You will write **one adapter file** and **one
config file**, both on this machine only. Read this whole guide before starting.

## Rules

1. **Do not modify modelrelay** (this repository). If something seems impossible without changing
   it, stop and report it (step 6).
2. **Never put company details in this repository**: URLs, field names, payloads, model names,
   credentials. They belong only in `~/.modelrelay/`.
3. **Never guess.** When a field is missing or a status is unknown, raise. Use `require()` for
   fields and `UnexpectedResponse` for unknown values. An error with the full payload is always
   better than a wrong answer.
4. **Never hide errors.** No bare `except`, no default values that cover up a missing field, no
   retry loops of your own. modelrelay already retries what can be retried.

## What modelrelay already does (do not reimplement)

Polling loop and backoff, retries on network errors/429/5xx, job timeout, token request and
renewal (at 28 of 30 minutes, plus retry on 401/403), SSE streaming for OpenAI-compatible APIs,
parsing OpenAI chat completions, events, trace ids, logs and error context.

## Where things live

```
~/.modelrelay/                      (C:\Users\<user>\.modelrelay\ on Windows)
├── config.toml                     URLs, credentials, model names, options
└── adapters/
    └── gateway.py                  class GatewayJobs: translates the job API's contract
```

Projects only `pip install` modelrelay; they contain nothing from the gateway.

## Step 1: create the files

```bash
pip install -e path/to/modelrelay     # or: pip install git+https://github.com/naruminho/modelrelay
modelrelay init --template gateway
```

This creates `~/.modelrelay/config.toml` and `~/.modelrelay/adapters/gateway.py`, full of TODOs.

## Step 2: discover the real contract

Write a throwaway script (keep it outside this repository) that calls the real endpoints with
`httpx` and prints the JSON:

1. the token endpoint: request fields, where the token is in the response;
2. the job start endpoint for the LLM text workflow: request body, where the job id is;
3. the job status endpoint, polled until it finishes: **every** status value you see, where the
   result is, and what an error looks like (send an invalid model name to get one);
4. the same for an image-generation request, if images are needed;
5. whether the start endpoint accepts a list of messages with roles, tool definitions, and files.

Save redacted examples of each payload next to the adapter (e.g. `~/.modelrelay/adapters/notes.md`)
so later changes are easy.

## Step 3: fill `config.toml`

- `[auth_options]`: `token_url`, `client_id`, `client_secret`, and the field names:
  `id_field` / `secret_field` (request), `token_field` (response, dotted path like `data.token`),
  `request_format` (`json` or `form`), `ttl_minutes`, `extra_fields` (fixed extra request fields).
- `[transports."gateway:GatewayJobs"] base_url`: root of the job API.
- `[transports.openai_compatible] base_url`: root of the OpenAI-compatible proxy, if there is one.
  Add `extra_headers = {...}` there if the proxy needs headers.
- `[models]`: model names used in code = exact names the gateway expects (with region prefixes).
- `verify_ssl = false`, or better `ca_bundle = "path/to/company-ca.pem"`, if TLS fails.
- `max_payload_mb`: the gateway's request size limit.

Check it with `modelrelay show` (secrets are masked).

## Step 4: fill `adapters/gateway.py`

| Method | Return |
|---|---|
| `submit_url(req)` | URL that starts a job (use `self.base_url`) |
| `poll_url(job_id, req)` | URL that returns the job's status/result |
| `build_submit(req)` | request body in the gateway's format |
| `parse_submit(data)` | the job id: `require(data, "path.to.id")` |
| `parse_poll(data)` | `JobState(status, response=..., error=..., raw=data)` with status in `queued`, `running`, `done`, `error` |
| `to_response(data)` | the result as a `Response` |

**Messages** (`req.messages`) are OpenAI-style. If the gateway accepts the same format, pass them
through. If it only accepts one prompt string, convert them explicitly (for example
`"system: ...\nuser: ...\nassistant: ..."`). If it cannot take something (files, a role), raise
`ModelRelayError("The job API does not accept ...")`. Never drop it silently.

**Result.** If the result is an OpenAI chat completion, use `parse_completion(...)`. Otherwise build
`Response(text=..., images=[Image.from_url(...)], usage=Usage(...), raw=data)` yourself.

**Tool calls.**
- The gateway accepts `tools` and returns OpenAI `tool_calls`: pass `req.tools` through and use
  `parse_completion`.
- It returns tool calls in another shape: build them with
  `modelrelay.tools.make_tool_call(id, name, raw_arguments_json)` into `Response(tool_calls=[...])`.
- It has no tool support: set `tools_mode = "emulated"` under `[transports."gateway:GatewayJobs"]`.
  modelrelay then describes the tools in the prompt and parses the answer. The adapter only has
  to return the model's raw text **unchanged**.

**Unusual token endpoint.** If `client_credentials` options are not enough, add to the same file:

```python
from modelrelay import ClientCredentials

class GatewayAuth(ClientCredentials):
    def fetch_token(self) -> str:
        ...  # call the endpoint with self.http; raise AuthError on any problem
```

and set `auth = "gateway:GatewayAuth"` in `config.toml` (keep the `[auth_options]`).

`src/modelrelay/testing/mock_adapter.py` in this repository is a complete, working adapter for a
fake gateway with nested payloads. Use it as the reference.

## Step 5: validate

```bash
set MODELRELAY_LOG=debug            # Windows cmd; PowerShell: $env:MODELRELAY_LOG="debug"
python examples/smoke_test.py --profile config --text-model <name> --vision-model <name> --image-model <name>
```

Then run `python examples/basics.py --model <name> --image-model <name>`: it is written like a
real project and must print all six sections without errors (section 6 shows an error on purpose).

Run the smoke test twice: once as is (`chat()` through jobs, `stream()` through the proxy), and once with
`stream_transport = "gateway:GatewayJobs"` to check streaming over jobs too. Every check must
print `OK`. Then test a long streamed answer through the proxy (ask for ~3000 words) and write
down whether and when it gets cut. A cut must raise `StreamInterrupted`, not return a truncated
answer.

| Error | Usually means |
|---|---|
| `UnexpectedResponse: Field 'x.y' not found` | wrong path in `require()`; the full payload is in `error.body` |
| `UnexpectedResponse: Unknown ... state` | a status missing from `STATUS` |
| `AuthError` | token endpoint fields or credentials |
| `ProviderError: HTTP 401/403` after a renewal | credentials lack permission, or the token goes in another header |
| `ProviderError: Network error` | URL, proxy or TLS (`ca_bundle` / `verify_ssl`) |
| `InvalidToolCall` | the model produced bad tool arguments; try another model or `tools_mode` |
| `JobTimeout` | the job never finished; raise `max_wait_seconds` or check the status mapping |

## Step 6: report

Report to the user:

- which smoke test checks passed, with the config used (`modelrelay show` output, secrets masked);
- the answers you found: tool support (native/emulated), message format, image generation,
  long-stream behavior over the proxy;
- anything that seems to need a change **in modelrelay itself**, described in generic words only
  (for example "the status endpoint returns a list instead of an object"), with no company
  details. Do not make that change yourself.
