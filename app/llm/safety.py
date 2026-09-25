"""Guards against an LLM provider's own failure text leaking through as if
it were a real answer/translation/summary.

Extracted out of `app.services.chat_service.ChatService` (Phase 1 of the
god-object split) -- these functions have zero dependency on chat state.
`ChatService` keeps its original private method names as one-line
delegators to these.
"""
import structlog

from app.llm.base import LLMResponse

log = structlog.get_logger(__name__)

# Best-effort detection of a provider error surfacing through `LLMProvider
# .stream()`, used only in `answer_stream()`'s live-streaming branch. Unlike
# `.chat()`, `.stream()` has no structured error field (see `safe_llm_text`
# for the non-streaming version of this problem) -- providers instead yield
# a friendly failure message as plain text (e.g. `GeminiProvider.stream`'s
# "Gemini API is unreachable...", `OllamaProvider`'s "Local LLM is
# unavailable..."), observed to always arrive as the STREAM'S FIRST AND ONLY
# chunk on failure. Checking only the first chunk against known phrasings is
# a deliberate simplification, not a guarantee -- a genuine answer starting
# with one of these exact phrases is essentially impossible, but a provider
# whose error wording isn't in this list would still leak through.
_LLM_ERROR_PHRASES = (
    "is unreachable", "is currently unavailable", "is not configured", "was rejected",
    "was not found", "rate limit or quota", "failed unexpectedly", "please start ollama",
    "would you like to download it",
)


def safe_llm_text(llm_response: LLMResponse, fallback: str) -> str:
    """Guards every non-streaming LLM call site against a provider error
    leaking through as if it were real content.

    `LLMProvider.chat()` surfaces a failed call (network unreachable, bad
    API key, rate limit, etc.) as a friendly string in `LLMResponse.content`
    rather than raising -- the caller has to check `.error` explicitly to
    tell the two apart. Every call site in this file used to check only
    for *empty* content before falling back, which let a real provider
    error (never empty) through untouched as if it were a genuine answer/
    translation/summary -- e.g. "Gemini API is unreachable. Check your
    network connection and try again." shown to the user as the answer to
    a legal question, disclaimer and all. Falls back to `fallback` (same
    as each site's pre-existing empty-content behavior) on error too.
    """
    if llm_response.error:
        log.warning("llm_call_failed", error=llm_response.error, provider=llm_response.provider)
        return fallback
    return llm_response.content.strip() or fallback


def failure_category(error: str | None) -> str | None:
    """Map internal/provider wording to a stable, non-sensitive label."""
    if not error:
        return None
    lowered = error.lower()
    if "deadline" in lowered or "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "rate limit" in lowered or "quota" in lowered or "429" in lowered:
        return "rate_limit"
    if any(marker in lowered for marker in ("401", "403", "api key", "rejected", "auth")):
        return "provider_auth"
    if "not configured" in lowered:
        return "provider_not_configured"
    return "provider_unavailable"


def looks_like_llm_error(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _LLM_ERROR_PHRASES)
