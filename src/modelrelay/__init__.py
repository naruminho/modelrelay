"""modelrelay: one small API to call LLMs anywhere.

    from modelrelay import llm
    print(llm.chat("Hello!", model="gpt-4o").text)
"""

from ._util import enable_logging, require
from .auth import BearerAuth, ClientCredentials, StaticToken, TokenProvider
from .config import Config
from .errors import (
    AuthError,
    ConfigError,
    InvalidToolCall,
    JobFailed,
    JobTimeout,
    ModelRelayError,
    PayloadTooLarge,
    ProviderError,
    StreamInterrupted,
    UnexpectedResponse,
)
from .relay import Relay
from .transports import JobState, JobsTransport, OpenAICompatible, Transport
from .types import ChatRequest, Event, Image, Response, ToolCall, Usage

__version__ = "0.1.0"


class _DefaultRelay:
    """`llm`: a Relay created on first use from the default config."""

    _relay: Relay | None = None

    def __getattr__(self, name):
        if _DefaultRelay._relay is None:
            _DefaultRelay._relay = Relay()
        return getattr(_DefaultRelay._relay, name)


llm = _DefaultRelay()

__all__ = [
    "AuthError",
    "BearerAuth",
    "ChatRequest",
    "ClientCredentials",
    "Config",
    "ConfigError",
    "Event",
    "Image",
    "InvalidToolCall",
    "JobFailed",
    "JobState",
    "JobTimeout",
    "JobsTransport",
    "ModelRelayError",
    "OpenAICompatible",
    "PayloadTooLarge",
    "ProviderError",
    "Relay",
    "Response",
    "StaticToken",
    "StreamInterrupted",
    "TokenProvider",
    "ToolCall",
    "Transport",
    "UnexpectedResponse",
    "Usage",
    "enable_logging",
    "llm",
    "require",
]
