import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import structlog
else:
    # structlog is a declared dependency; `None` here lets `configure_logging()`
    # fall back to stdlib logging on a host missing it rather than failing to
    # import. Type checking runs against the real module.
    try:
        import structlog
    except ModuleNotFoundError:
        structlog = None

from app.core.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )
    _silence_secret_leaking_loggers()
    if structlog is None:
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Loggers that write full request URLs at INFO. Gemini authenticates with the
# API key as a QUERY PARAMETER (`?key=...`), so httpx's own one-line request
# log put the live secret into stdout on every single call:
#
#   HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/
#   models/gemma-4-26b-a4b-it:generateContent?key=AQ.Ab8RN6Iv... "200 OK"
#
# Confirmed in this app's real log output while reproducing a drafting
# failure. Anywhere those logs are shipped, retained, or pasted into a bug
# report, the key goes with them. The app's own structured logging never logs
# a URL, so raising these to WARNING loses nothing we rely on: transport
# failures are already reported by `GeminiProvider`'s own `gemini_http_error`
# / `gemini_unavailable` events, which log a status code and an error type,
# never a URL.
_URL_LOGGING_LIBRARIES = ("httpx", "httpcore", "openai", "anthropic", "google", "urllib3")


def _silence_secret_leaking_loggers() -> None:
    for name in _URL_LOGGING_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
