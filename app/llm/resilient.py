"""Retry + provider-failover wrapper around `LLMProvider`.

Why this exists (draft-quality incident, Sept 2026): every provider in this
codebase reports a failure by RETURNING an `LLMResponse` whose `error` is
set and whose `content` holds a human-readable error sentence -- they do not
raise (see `LLMResponse.error` in `app/llm/base.py`). That contract is fine
for the chat path, which checks `error` and shows the sentence.

`LegalDraftEngine._render_sections` did not check it. A transient provider
outage therefore came back as an `LLMResponse` whose content was
"Gemini API is currently unavailable. Please try again.", which
`_parse_sections` correctly found no "## " headings in, returned `{}` for,
and which silently routed generation into the deterministic fallback. The
user was handed a one-page skeleton -- presented with no warning as their
finished legal document -- because a network call blipped. Neither a retry
nor a second provider was ever attempted.

This wrapper closes that hole for any caller that wants it:

* retries a failed call with exponential backoff, because most provider
  failures in practice (429 rate limit, 503, connection reset) succeed on a
  second attempt seconds later;
* then fails over to configured backup providers in order, so a Gemini quota
  exhaustion falls to Groq/OpenAI/Ollama rather than to a stub document;
* returns the LAST error response unchanged when everything is exhausted, so
  callers still see a normal `LLMResponse` with `error` set and can decide
  for themselves how to degrade -- this wrapper never invents content and
  never converts a failure into a fake success.

Deliberately NOT retried: a response that arrives successfully but is
unusable (wrong format, too short). That is a prompt/model-quality problem,
not a transport problem, and it is handled where the semantics are known --
in `LegalDraftEngine`, which knows what "unusable" means for a draft.
"""

import asyncio
import re
import time
from collections.abc import AsyncIterator

import structlog

from app.llm.base import ChatMessage, LLMProvider, LLMResponse
from app.llm.deadline import MIN_VIABLE_CALL_SECONDS, has_time_for, log_exhausted, remaining_seconds

log = structlog.get_logger(__name__)

# Some "thinking" models put their chain-of-thought inline in the normal text
# as literal `<think>...</think>` markup instead of (or in addition to) using
# a provider API's structured "thought" flag -- confirmed live with the
# configured Gemini model (`gemma-4-26b-a4b-it`), whose own provider already
# filters the structured `"thought": true` parts (see `GeminiProvider.
# _parse_response`) but still let raw reasoning -- including verbatim
# fragments of the system prompt's own rules -- leak straight into the
# user-facing answer when the model wrote it as inline tags instead. Handled
# here, once, for every provider (Gemini/Groq/OpenAI/Claude/DeepSeek/Ollama),
# since it's a model-output quirk, not something specific to one HTTP client.
_INLINE_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# An unterminated `<think>` (e.g. truncated at a length cap) means everything
# from that point on is still reasoning, not an answer -- drop it too rather
# than showing a half-finished internal monologue.
_UNTERMINATED_THINK_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)


def _strip_inline_thinking(content: str) -> str:
    if not content or "<think" not in content.lower():
        return content
    cleaned = _INLINE_THINK_BLOCK_RE.sub("", content)
    cleaned = _UNTERMINATED_THINK_RE.sub("", cleaned)
    return cleaned.strip()


class _InlineThinkStreamFilter:
    """Incrementally strips `<think>...</think>` from a live token stream.

    `_strip_inline_thinking` above is correct but only for a COMPLETE string
    -- it cannot be applied to the streaming path by running it once per
    chunk and yielding the result, because by the time a chunk contains
    `</think>` the reasoning text before it has *already* been sent to the
    client as earlier "token" events. Stripping only from the final
    assembled string (as `answer_stream`'s "done" payload does) leaves the
    live stream itself leaking every token of reasoning the moment it
    arrives -- confirmed live: the exact same `<think>...Rule 2's exact
    fallback line...</think>` leak reproduced via the non-streaming path
    also reproduces token-by-token over `/chat/stream` with only the
    non-streaming fix applied.

    This holds back only the minimum needed to recognise a tag that may be
    split across two chunks (at most `len("</think>") - 1` = 7 trailing
    characters), so ordinary text -- including legitimate angle brackets
    that are not literally "<think" -- passes through with effectively no
    added latency. Nothing inside a `<think>` block, complete or not, is
    ever handed to `feed()`'s caller.
    """

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"
    _MAX_HOLDBACK = max(len(_OPEN_TAG), len(_CLOSE_TAG)) - 1

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, piece: str) -> str:
        """Returns the portion of `piece` (plus any previously held-back
        text) that is now confirmed to be outside a think block. May be
        `""` if everything so far is still reasoning or an unresolved
        partial tag."""
        if not piece:
            return ""
        self._buffer += piece
        emitted: list[str] = []
        while True:
            if self._in_think:
                idx = self._buffer.lower().find(self._CLOSE_TAG)
                if idx == -1:
                    self._buffer = self._buffer[-self._MAX_HOLDBACK :]
                    return "".join(emitted)
                self._buffer = self._buffer[idx + len(self._CLOSE_TAG) :]
                self._in_think = False
                continue
            idx = self._buffer.lower().find("<think")
            if idx == -1:
                if len(self._buffer) > self._MAX_HOLDBACK:
                    emitted.append(self._buffer[: -self._MAX_HOLDBACK])
                    self._buffer = self._buffer[-self._MAX_HOLDBACK :]
                return "".join(emitted)
            close_bracket = self._buffer.find(">", idx)
            if close_bracket == -1:
                # "<think" has started arriving but the tag isn't complete
                # yet (e.g. this chunk ended exactly at "<think"). Emit
                # everything before it; hold the rest for the next piece.
                emitted.append(self._buffer[:idx])
                self._buffer = self._buffer[idx:]
                return "".join(emitted)
            tag = self._buffer[idx : close_bracket + 1]
            if tag.lower() != self._OPEN_TAG:
                # Started with "<think" but isn't the exact tag (e.g. some
                # other markup) -- ordinary text, not a reasoning marker.
                emitted.append(self._buffer[: close_bracket + 1])
                self._buffer = self._buffer[close_bracket + 1 :]
                continue
            emitted.append(self._buffer[:idx])
            self._buffer = self._buffer[close_bracket + 1 :]
            self._in_think = True

    def flush(self) -> str:
        """Called once the underlying stream ends. Whatever is still
        buffered while genuinely mid-`<think>` is unterminated reasoning
        (e.g. truncated at a length cap) and must never reach the caller;
        anything buffered while NOT mid-think (just a short tail that never
        turned out to start a tag) is ordinary text and is released now."""
        if self._in_think:
            self._buffer = ""
            return ""
        remaining, self._buffer = self._buffer, ""
        return remaining


# `error_kind` values that are worth another attempt. Preferred over the
# substring scan below, which stays only as a fallback for providers that do
# not yet set `error_kind` (every provider except Gemini today).
_RETRYABLE_ERROR_KINDS = {"timeout", "rate_limit", "http_error", "provider_error"}
# Never retried, whatever the wording: these fail identically every time, so a
# retry only spends budget that graceful degradation needs. Failover to the
# next PROVIDER is still the useful move and still happens.
_NON_RETRYABLE_ERROR_KINDS = {"auth", "not_found", "not_configured", "deadline_exhausted"}

# Errors worth another attempt: transient transport/capacity problems. A
# rejected API key or a missing model will fail identically every time, so
# retrying those only adds latency to an outcome that is already decided --
# failover to the next PROVIDER is the useful move there, and still happens.
_RETRYABLE_MARKERS = (
    "unavailable", "rate limit", "quota", "timeout", "timed out",
    "connection", "temporarily", "try again", "503", "502", "504", "429",
)


# Non-retryable wordings, checked before the retryable markers below: a
# rejected key ("was rejected") or a missing model ("was not found") contains
# no retryable marker today, but relying on that absence is fragile -- naming
# them explicitly means a reworded message can never turn a permanent 4xx
# failure into three pointless retries.
_NON_RETRYABLE_MARKERS = (
    "was rejected", "not configured", "was not found", "invalid", "permission", "forbidden",
    "401", "403", "404", "400",
)


def _is_retryable(response: LLMResponse) -> bool:
    """Whether `response`'s failure is worth another attempt.

    Prefers the structured `error_kind` and falls back to scanning the
    human-readable sentence for providers that do not set one yet.
    """
    if response.error_kind:
        if response.error_kind in _NON_RETRYABLE_ERROR_KINDS:
            return False
        return response.error_kind in _RETRYABLE_ERROR_KINDS
    lowered = (response.error or "").lower()
    if any(marker in lowered for marker in _NON_RETRYABLE_MARKERS):
        return False
    return any(marker in lowered for marker in _RETRYABLE_MARKERS)


class ResilientLLMProvider(LLMProvider):
    """Wraps a primary provider plus ordered fallbacks.

    Constructed with already-built providers rather than provider names so it
    stays trivially testable (pass two fakes) and so `LLMFactory` keeps sole
    ownership of how a name becomes a provider.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider] | None = None,
        *,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 0.75,
    ) -> None:
        self._providers = [primary, *(fallbacks or [])]
        self.provider_name = getattr(primary, "provider_name", "resilient")
        self.max_attempts = max(1, max_attempts)
        self.initial_backoff_seconds = initial_backoff_seconds

    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        """Retry + failover, bounded by the request-wide deadline.

        Previously this was the single biggest source of unbounded latency in
        the app: `max_attempts` (3) x `settings.llm_timeout` (120s) on the
        primary alone is ~360s of server work for a request the Streamlit
        client abandons at 180s. Confirmed live -- see the timeline in
        `app/llm/deadline.py`. Every attempt now checks the remaining budget
        first, and both the attempt and its backoff are skipped when there
        isn't enough left to be worth it.
        """
        last_response: LLMResponse | None = None
        attempts_made = 0
        started = time.monotonic()
        for provider_index, provider in enumerate(self._providers):
            provider_name = getattr(provider, "provider_name", "unknown")
            # Only the primary is retried in place. A fallback is already the
            # second chance; retrying each of them three times too would turn
            # one slow request into a multi-minute stall for the user.
            attempts = self.max_attempts if provider_index == 0 else 1
            for attempt in range(attempts):
                if not has_time_for():
                    log_exhausted(
                        "resilient_chat", provider=provider_name, attempt=attempt + 1,
                        attempts_made=attempts_made,
                    )
                    return last_response or LLMResponse(
                        content="The assistant is taking too long to respond. Please try again.",
                        model=getattr(provider, "model", ""), provider=provider_name,
                        error="Deadline exhausted before any provider could be tried.",
                        error_kind="deadline_exhausted",
                    )
                try:
                    response = await provider.chat(messages, temperature=temperature)
                except Exception as exc:  # noqa: BLE001 - a provider that raises must not kill the caller
                    log.warning(
                        "llm_provider_raised",
                        provider=provider_name,
                        attempt=attempt + 1,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    last_response = LLMResponse(
                        content=str(exc),
                        model=getattr(provider, "model", ""),
                        provider=provider_name,
                        error=str(exc),
                        error_kind="provider_error",
                    )
                    response = last_response
                attempts_made += 1
                if not response.error:
                    response.content = _strip_inline_thinking(response.content)
                    if not response.content.strip():
                        # The model produced nothing but reasoning -- not a
                        # transport failure, but just as unusable to the
                        # caller. Marking it as a retryable error reuses the
                        # existing retry/failover loop below rather than
                        # handing back an empty "successful" answer.
                        response.error = "Model returned only internal reasoning with no answer text."
                        response.error_kind = "provider_error"
                    else:
                        response.retry_count = max(response.retry_count, attempts_made - 1)
                        return response
                last_response = response
                is_last_attempt_on_provider = attempt == attempts - 1
                if is_last_attempt_on_provider or not _is_retryable(response):
                    break
                backoff = self.initial_backoff_seconds * (2**attempt)
                left = remaining_seconds()
                if left is not None and left <= backoff + MIN_VIABLE_CALL_SECONDS:
                    log_exhausted("resilient_backoff", provider=provider_name, attempt=attempt + 1)
                    break
                await asyncio.sleep(backoff)
            log.warning(
                "llm_provider_failed_over",
                provider=provider_name,
                error_type=(last_response.error_kind if last_response else None),
                attempts_made=attempts_made,
                elapsed_seconds=round(time.monotonic() - started, 2),
                remaining_seconds=(round(remaining, 2) if (remaining := remaining_seconds()) is not None else None),
                remaining_fallbacks=len(self._providers) - provider_index - 1,
            )
        # Everything exhausted. Hand back the real error, unchanged -- the
        # caller decides how to degrade; this layer never fabricates content.
        return last_response or LLMResponse(
            content="No LLM provider is configured.", model="", provider="none",
            error="No LLM provider is configured.", error_kind="not_configured",
        )

    async def stream(self, messages: list[ChatMessage], temperature: float = 0.1) -> AsyncIterator[str]:
        """Streams from the primary only.

        Mid-stream failover would mean replaying tokens the user has already
        seen; the non-streaming `chat()` path above is what the drafting
        engine uses and is where resilience actually matters.

        Every yielded chunk passes through `_InlineThinkStreamFilter` first
        (see its docstring for why a post-hoc strip on the assembled text is
        NOT sufficient here): callers of `stream()` (chiefly `ChatService.
        answer_stream`, `/chat/stream`) forward each yielded piece straight
        to the client as it arrives, so filtering has to happen before a
        chunk is yielded, not after the stream ends.
        """
        think_filter = _InlineThinkStreamFilter()
        async for chunk in self._providers[0].stream(messages, temperature=temperature):
            cleaned = think_filter.feed(chunk)
            if cleaned:
                yield cleaned
        tail = think_filter.flush()
        if tail:
            yield tail

    async def health(self) -> bool:
        for provider in self._providers:
            try:
                if await provider.health():
                    return True
            except Exception as exc:  # noqa: BLE001 - one bad provider must not mask a healthy fallback
                log.warning(
                    "llm_provider_health_check_failed",
                    provider=getattr(provider, "provider_name", "unknown"),
                    error=str(exc),
                )
        return False
