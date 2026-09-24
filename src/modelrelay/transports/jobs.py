from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

from .._util import log, require
from ..errors import JobFailed, JobTimeout, ModelRelayError, ProviderError, UnexpectedResponse
from ..types import ChatRequest, Event, Response
from .base import Transport
from .openai_format import parse_completion

JOB_STATUSES = ("queued", "running", "done", "error")


@dataclass
class JobState:
    status: str  # one of JOB_STATUSES
    response: Response | None = None
    error: str | None = None
    raw: object = None  # the poll payload, kept for errors and debugging


class JobsTransport(Transport):
    """Generic submit + poll transport for gateways that queue work.

    Flow: POST the request -> get a job id -> GET the job until it is done.
    Polling waits `poll_interval` seconds, growing by `poll_backoff` up to
    `poll_max_interval`. Network errors, 429 and 5xx while polling are retried up
    to `max_poll_errors` times in a row, keeping the same job; each retry is logged.

    The defaults speak a simple contract (see `build_submit` / `parse_poll`).
    For a real gateway, subclass and override the hooks; that subclass is your
    private adapter. Hooks must raise (e.g. with `require()`) instead of guessing
    when the payload is not what they expect.
    """

    supports_text_stream = False

    def __init__(self, http, auth, config, options=None, *, sleep: Callable = time.sleep, clock: Callable = time.monotonic):
        super().__init__(http, auth, config, options)
        o = self.options
        self.poll_interval = float(o.get("poll_interval", 0.5))
        self.poll_max_interval = float(o.get("poll_max_interval", 5.0))
        self.poll_backoff = float(o.get("poll_backoff", 1.5))
        self.max_wait_seconds = float(o.get("max_wait_seconds", 900))
        self.max_poll_errors = int(o.get("max_poll_errors", 5))
        self._sleep = sleep
        self._clock = clock

    # ---- hooks: override these in an adapter ------------------------------------

    def submit_url(self, req: ChatRequest) -> str:
        return f"{self.base_url}/jobs"

    def poll_url(self, job_id: str, req: ChatRequest) -> str:
        return f"{self.base_url}/jobs/{job_id}"

    def build_submit(self, req: ChatRequest) -> dict:
        body = {"model": req.model, "messages": req.messages, **req.params}
        if req.tools:
            body["tools"] = req.tools
        return body

    def parse_submit(self, data: dict) -> str:
        return str(require(data, "job_id", "job id"))

    def parse_poll(self, data: dict) -> JobState:
        status = require(data, "status", "job status")
        if status == "done":
            return JobState("done", response=parse_completion(require(data, "result")), raw=data)
        if status == "error":
            return JobState("error", error=str(data.get("error")), raw=data)
        return JobState(status, raw=data)

    # ---- flow ------------------------------------------------------------------

    def complete(self, req: ChatRequest) -> Response:
        for event in self.stream(req):
            if event.type == "done":
                return event.response
        raise ProviderError("Job ended without a result", trace_id=req.trace_id)  # pragma: no cover

    def stream(self, req: ChatRequest) -> Iterator[Event]:
        start = self._clock()
        data = self.send("POST", self.submit_url(req), req, self.body_for(self.build_submit(req)))
        job_id = self.parse_submit(data)
        log.info("%s job %s submitted", req.trace_id, job_id)
        yield Event("queued", job_id=job_id, trace_id=req.trace_id)

        interval = self.poll_interval
        errors = 0
        last_status = None
        while True:
            elapsed = self._clock() - start
            if elapsed > self.max_wait_seconds:
                raise JobTimeout(
                    f"Job still not done after {elapsed:.0f}s (max_wait_seconds={self.max_wait_seconds:.0f})",
                    job_id=job_id, last_status=last_status, trace_id=req.trace_id,
                )
            self._sleep(interval)
            interval = min(interval * self.poll_backoff, self.poll_max_interval)
            try:
                data = self.send("GET", self.poll_url(job_id, req), req)
            except ProviderError as e:
                if not e.retriable:
                    e.context.setdefault("job_id", job_id)
                    raise
                errors += 1
                log.warning("%s job %s poll failed (%d/%d), retrying: %s",
                            req.trace_id, job_id, errors, self.max_poll_errors, e)
                if errors > self.max_poll_errors:
                    e.context.update(job_id=job_id, poll_errors=errors)
                    raise
                continue
            errors = 0

            try:
                state = self.parse_poll(data)
            except ModelRelayError as e:
                e.context.setdefault("job_id", job_id)
                raise
            if state.status not in JOB_STATUSES:
                raise UnexpectedResponse(
                    f"parse_poll returned unknown status {state.status!r}; expected one of {JOB_STATUSES}",
                    body=data, job_id=job_id, trace_id=req.trace_id,
                )
            if state.status != last_status:
                log.info("%s job %s -> %s (%.1fs)", req.trace_id, job_id, state.status, self._clock() - start)
                last_status = state.status
            elapsed = self._clock() - start
            if state.status == "done":
                if state.response is None:
                    raise UnexpectedResponse("Job is done but has no response", body=data, job_id=job_id,
                                             trace_id=req.trace_id)
                yield Event("done", response=state.response, elapsed=elapsed, job_id=job_id, trace_id=req.trace_id)
                return
            if state.status == "error":
                raise JobFailed(f"Job failed: {state.error}", body=state.raw, job_id=job_id,
                                trace_id=req.trace_id, elapsed=round(elapsed, 2))
            yield Event(state.status, elapsed=elapsed, job_id=job_id, trace_id=req.trace_id)
