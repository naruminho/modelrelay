class ModelRelayError(Exception):
    """Base class for every error raised by modelrelay.

    `context` carries what you need to trace the problem: trace_id, transport,
    method, url, status, job_id, elapsed... It is printed with the message.
    """

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.message = message
        self.context = {k: v for k, v in context.items() if v is not None}

    @property
    def trace_id(self) -> str | None:
        return self.context.get("trace_id")

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message} [{details}]"


class ConfigError(ModelRelayError):
    """The configuration is missing or invalid."""


class AuthError(ModelRelayError):
    """Could not obtain a token."""


class PayloadTooLarge(ModelRelayError):
    """The request is bigger than `max_payload_mb`."""


class ProviderError(ModelRelayError):
    """The provider answered with an error, or with something we could not understand.

    `status` is the HTTP status (None for network errors); `body` is the full,
    untruncated response body.
    """

    def __init__(self, message: str, status: int | None = None, body: object = None, **context):
        super().__init__(message, status=status, **context)
        self.status = status
        self.body = body

    @property
    def retriable(self) -> bool:
        return self.status is None or self.status >= 500 or self.status == 429


class UnexpectedResponse(ProviderError):
    """The response does not have the shape the transport/adapter expects (contract changed?).

    Never retried: repeating the call would not change the contract.
    """

    @property
    def retriable(self) -> bool:
        return False


class StreamInterrupted(ProviderError):
    """The stream ended before the provider said it was finished. `partial` has what arrived."""

    def __init__(self, message: str, partial=None, **context):
        super().__init__(message, **context)
        self.partial = partial


class JobFailed(ProviderError):
    """A job finished with an error status."""


class JobTimeout(ModelRelayError):
    """A job did not finish within `max_wait_seconds`."""


class InvalidToolCall(ModelRelayError):
    """The model asked for a tool with arguments that are not valid JSON.

    `response` is the full Response and `raw` the offending text, so you can
    send the problem back to the model if you want to.
    """

    def __init__(self, message: str, raw: str = "", response=None, **context):
        super().__init__(message, **context)
        self.raw = raw
        self.response = response
