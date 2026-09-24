"""Adapter for a job-based gateway (submit a request, then poll until it is done).

This file lives in ~/.modelrelay/adapters/ on this machine only. Never commit it to a
public repository. The config refers to it as "gateway:GatewayJobs".

modelrelay already does the polling loop, backoff, retries, timeouts, token renewal and
error reporting. This class only translates the gateway's contract. Fill in every TODO.
Rules (see ADAPTER_GUIDE.md in the modelrelay repository):
  - never guess: use require() for fields and raise UnexpectedResponse for anything unknown;
  - never catch and hide errors here;
  - keep the full payload in JobState.raw.
"""

from __future__ import annotations

from modelrelay import (
    ChatRequest,
    Image,
    JobsTransport,
    JobState,
    Response,
    UnexpectedResponse,
    Usage,
    require,
)
from modelrelay.transports import parse_completion

# TODO: every status the gateway can return -> "queued" | "running" | "done" | "error".
# Anything missing here raises UnexpectedResponse, on purpose.
STATUS: dict[str, str] = {
    # "STARTED": "queued",
    # "IN_PROGRESS": "running",
    # "FINISHED": "done",
    # "FAILED": "error",
}


class GatewayJobs(JobsTransport):
    # ---- where to send ----------------------------------------------------------------

    def submit_url(self, req: ChatRequest) -> str:
        # self.base_url comes from [transports."gateway:GatewayJobs"] base_url in config.toml
        raise NotImplementedError("TODO: URL of the endpoint that starts a job")

    def poll_url(self, job_id: str, req: ChatRequest) -> str:
        raise NotImplementedError("TODO: URL of the endpoint that returns the job status/result")

    # ---- what to send -----------------------------------------------------------------

    def build_submit(self, req: ChatRequest) -> dict:
        # req.model    model name, already mapped through [models]
        # req.messages OpenAI-style messages: [{"role": "system"|"user"|"assistant"|"tool", "content": ...}]
        #              content may be a list of parts (text, image_url, file) when files are attached
        # req.tools    OpenAI-style tool definitions, or None (always None with tools_mode = "emulated")
        # req.params   extra keyword arguments from chat()/stream() (temperature, max_tokens, ...)
        raise NotImplementedError("TODO: body of the start request, in the gateway's format")

    # ---- how to read the answers ------------------------------------------------------

    def parse_submit(self, data: dict) -> str:
        # Return the job/execution id from the start response.
        raise NotImplementedError("TODO: e.g. return require(data, 'data.execution.id', 'execution id')")

    def parse_poll(self, data: dict) -> JobState:
        raise NotImplementedError("TODO: see the example below")
        # Example:
        # gateway_status = require(data, "data.execution.state", "execution state")
        # if gateway_status not in STATUS:
        #     raise UnexpectedResponse(f"Unknown execution state {gateway_status!r}", body=data)
        # status = STATUS[gateway_status]
        # if status == "done":
        #     return JobState("done", response=self.to_response(data), raw=data)
        # if status == "error":
        #     return JobState("error", error=require(data, "data.output.error.message"), raw=data)
        # return JobState(status, raw=data)

    def to_response(self, data: dict) -> Response:
        """The finished job's result as a Response."""
        raise NotImplementedError("TODO: see the two options below")
        # Option 1: the result is an OpenAI chat completion ({"choices": [...]}):
        #     return parse_completion(require(data, "path.to.result"))
        #
        # Option 2: any other shape. Build the Response yourself:
        #     result = require(data, "path.to.result")
        #     return Response(
        #         text=require(result, "text"),
        #         images=[Image.from_url(u) for u in result.get("images", [])],  # data: URLs or links
        #         usage=Usage(input_tokens=result.get("input_tokens"), output_tokens=result.get("output_tokens")),
        #         raw=data,
        #     )
        # Tool calls: see ADAPTER_GUIDE.md ("Tool calls").
