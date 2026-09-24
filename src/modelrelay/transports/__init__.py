from .base import Transport, json_or_raise
from .jobs import JobState, JobsTransport
from .openai_compatible import OpenAICompatible
from .openai_format import StreamAccumulator, parse_completion

BUILTIN_TRANSPORTS = {
    "openai_compatible": OpenAICompatible,
    "jobs": JobsTransport,
}

__all__ = [
    "BUILTIN_TRANSPORTS",
    "JobState",
    "JobsTransport",
    "OpenAICompatible",
    "StreamAccumulator",
    "Transport",
    "json_or_raise",
    "parse_completion",
]
