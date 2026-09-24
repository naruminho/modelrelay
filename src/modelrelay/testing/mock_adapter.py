"""Example adapter for the mock gateway's job API.

This is the template for a private adapter: the real one lives in a separate,
private package and translates your gateway's contract the same way.

Rule of thumb: never guess. If a field is missing or a status is unknown, raise
(require() / UnexpectedResponse) so the problem shows up with the full payload.
"""

from __future__ import annotations

from .._util import require
from ..errors import UnexpectedResponse
from ..transports import JobState, JobsTransport, parse_completion
from ..types import ChatRequest

# Gateway status -> generic status
STATUS = {
    "STARTED": "queued",
    "IN_PROGRESS": "running",
    "FINISHED": "done",
    "FAILED": "error",
}


class MockJobsTransport(JobsTransport):
    def build_submit(self, req: ChatRequest) -> dict:
        inputs = {"model": req.model, "messages": req.messages, **req.params}
        if req.tools:
            inputs["tools"] = req.tools
        return {"workflow": "text-generation", "inputs": inputs}

    def parse_submit(self, data: dict) -> str:
        return require(data, "data.execution.id", "execution id")

    def parse_poll(self, data: dict) -> JobState:
        gateway_status = require(data, "data.execution.state", "execution state")
        if gateway_status not in STATUS:
            raise UnexpectedResponse(f"Unknown execution state {gateway_status!r}", body=data)
        status = STATUS[gateway_status]
        if status == "done":
            return JobState("done", response=parse_completion(require(data, "data.output.result")), raw=data)
        if status == "error":
            return JobState("error", error=require(data, "data.output.error.message", "error message"), raw=data)
        return JobState(status, raw=data)
