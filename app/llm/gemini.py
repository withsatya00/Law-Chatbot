import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx
import structlog

from app.core.config import settings
from app.llm.base import ChatMessage, LLMProvider, LLMResponse
from app.llm.deadline import MIN_VIABLE_CALL_SECONDS, clamp_timeout, has_time_for, log_exhausted

log = structlog.get_logger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Part 29: narrowed to genuinely transient/rate-limit conditions only. 500/502
# (generic server errors) used to be retried too, but they're just as likely
# to be a broken request as a transient blip -- retrying them ate into the
# interactive-chat latency budget for little real payoff. Still NOT
# 400/401/403/404 (bad request/key/model), which fail identically every time.
_RETRIABLE_STATUS_CODES = {429, 503, 504}

# Backoff schedule for retries 1/2/3 (Part 29 spec): 500ms, 1s, 2s. Total
# possible backoff sleep across all 3 retries is 3.5s, under the spec's <4s
# ceiling for retry sleep time specifically (separate from the request time
# itself, which the overall per-call timeout below bounds).
_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    # Shared across every PRODUCTION `GeminiProvider` instance -- i.e. every
    # one constructed without an explicit `transport=` (test instances always
    # pass one). Mirrors the existing `EmbeddingProvider._model_cache`
    # pattern elsewhere in this codebase: instances themselves stay cheap to
    # construct and fully independent for testing/mocking, but the one
    # genuinely expensive shared resource -- a pooled, keep-alive HTTP
    # connection -- is cached at the class level instead of rebuilt from
    # scratch on every single call. Before this, `chat()`/`stream()`/
    # `health()` each opened `async with httpx.AsyncClient(...)` fresh,
    # paying a full TCP+TLS handshake every time instead of reusing a warm
    # connection -- with `ChatService()`/its `GeminiProvider` rebuilt per
    # HTTP request too, that meant essentially zero connection reuse ever.
    _shared_client_cache: ClassVar[dict[tuple[Any, ...], httpx.AsyncClient]] = {}
    _shared_client_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float | None = None,
        overall_timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
        retry_backoff_seconds: tuple[float, ...] = _RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.gemini_model
        # `read_timeout_seconds`/`overall_timeout_seconds` default from
        # `settings.llm_timeout` when not explicitly passed, instead of the
        # old fixed 20s/25s (Part 29). Those fixed numbers were tuned for a
        # fast, non-"thinking" model and silently ignored `settings.llm_timeout`
        # entirely -- unlike `OllamaProvider`, which already reads it. Once
        # `GEMINI_MODEL` switched to one with mandatory "thinking" (~40-50s
        # per call even for a normal RAG-sized prompt, confirmed via direct
        # testing), every real request started blowing past the old fixed
        # 25s ceiling and getting killed client-side -- with the app-level
        # `llm_timeout` setting (120s) sitting there unused the entire time,
        # making it look like a broken provider rather than a too-short
        # internal timeout. `connect_timeout_seconds` stays a separate, short,
        # fixed default on purpose: a genuinely dead connection should fail
        # fast regardless of how long the caller is willing to wait for a
        # slow-but-working model.
        resolved_overall_timeout = (
            overall_timeout_seconds if overall_timeout_seconds is not None else float(settings.llm_timeout)
        )
        # Capped meaningfully below `resolved_overall_timeout` rather than
        # equal to it (was: the ENTIRE remaining overall timeout minus the
        # connect timeout -- effectively ~115s at the old 120s `llm_timeout`).
        # That sized a single attempt's read timeout so close to the whole
        # call's budget that `max_retries`/backoff above existed only on
        # paper: confirmed live, a request whose pooled keep-alive connection
        # had silently gone dead (no FIN/RST -- e.g. an idle-timing-out local
        # TLS-inspecting proxy) spent the FULL ~115s read timeout hung on
        # that one dead attempt, left too little of the request-wide deadline
        # for `_chat_with_retries`' own backoff-and-retry to run at all
        # ("gemini_retry_skipped_no_budget"), and only survived because
        # `ResilientLLMProvider` happened to still have enough of the OUTER
        # deadline left for a second, fresh `GeminiProvider.chat()` call.
        #
        # The cap itself must stay well above `GEMINI_MODEL`'s own real
        # generation time, not just above network-hang territory: this app's
        # configured model is a mandatory-"thinking" one kept specifically
        # for its much higher daily quota vs. the faster non-thinking
        # variants (confirmed: 16 requests/day vs. 1600), and direct
        # measurement against this app's actual RAG-sized prompt (system
        # prompt + up to 6 retrieved chunks) showed single legitimate calls
        # regularly taking 60-90s. A cap tuned only for "abandon a dead
        # connection quickly" (e.g. 45s) would abort those same legitimate,
        # still-working calls before they ever finish -- indistinguishable
        # from a hang, and worse than not capping at all, since it never
        # gets a chance to succeed on any attempt. 110s comfortably covers
        # the observed legitimate range while still leaving ~50s of
        # `resolved_overall_timeout` for a real retry if this attempt is
        # actually a dead connection rather than a slow-but-working one.
        resolved_read_timeout = (
            read_timeout_seconds if read_timeout_seconds is not None
            else max(5.0, min(110.0, resolved_overall_timeout - connect_timeout_seconds))
        )
        # Split connect/read timeouts (Part 29): a connection that never
        # establishes should fail fast rather than eating the same generous
        # budget as a slow-but-connected generation -- a single flat timeout
        # couldn't distinguish the two, so a dead network took just as long
        # to fail as a legitimately slow-but-working response.
        self._timeout = httpx.Timeout(
            connect=connect_timeout_seconds, read=resolved_read_timeout,
            write=resolved_read_timeout, pool=connect_timeout_seconds,
        )
        # Hard ceiling on the ENTIRE call, including every retry and backoff
        # sleep -- per-attempt timeouts alone don't bound total latency: 3
        # retries stacked at the read timeout each could otherwise run far
        # longer than any single caller should wait.
        self.overall_timeout_seconds = resolved_overall_timeout
        # Only ever set in tests, to inject an `httpx.MockTransport` -- `None`
        # in production means httpx uses its normal default transport, and
        # this instance participates in the shared client cache above.
        self._transport = transport
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._instance_client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._transport is not None:
            # Test-only path -- kept scoped to this one instance, never
            # shared, so each test's mock handler is fully isolated from
            # every other test's and from the production cache below.
            if self._instance_client is None:
                self._instance_client = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
            return self._instance_client
        cache_key = (self.api_key, self._timeout.connect, self._timeout.read)
        cached = GeminiProvider._shared_client_cache.get(cache_key)
        if cached is not None:
            return cached
        with GeminiProvider._shared_client_lock:
            cached = GeminiProvider._shared_client_cache.get(cache_key)
            if cached is None:
                cached = httpx.AsyncClient(timeout=self._timeout)
                GeminiProvider._shared_client_cache[cache_key] = cached
            return cached

    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        if not self.api_key:
            return self._error_response("Gemini API key is not configured.")
        # The per-call ceiling is now the SMALLER of this provider's own
        # configured timeout and whatever is left on the request-wide
        # deadline (`app/llm/deadline.py`). Before this, a 120s call could be
        # started with 20s left before the caller's HTTP client gave up --
        # confirmed live: `gemini_overall_timeout` fired 24s AFTER Streamlit
        # had already shown the user a transport error.
        if not has_time_for():
            log_exhausted("gemini_chat", provider=self.provider_name, model=self.model)
            return self._error_response(
                "Gemini API is currently unavailable. Please try again.", error_kind="deadline_exhausted"
            )
        effective_timeout = clamp_timeout(self.overall_timeout_seconds)
        started = time.monotonic()
        try:
            return await asyncio.wait_for(
                self._chat_with_retries(messages, temperature, effective_timeout), timeout=effective_timeout
            )
        except TimeoutError:
            log.warning(
                "gemini_overall_timeout",
                timeout_seconds=effective_timeout,
                configured_timeout_seconds=self.overall_timeout_seconds,
                elapsed_seconds=round(time.monotonic() - started, 2),
                model=self.model,
            )
            # Post-Phase-3 hardening (Phase 2, milestone C): this used to say
            # "Gemini API is unreachable. Check your network connection and try
            # again." That is a different failure, and saying it here sent
            # every reader after the wrong thing.
            #
            # Confirmed live on this machine while producing a consumer
            # complaint: `scripts/validate_environment.py` reported the
            # provider healthy seconds earlier, and this branch still fired --
            # because a short health probe finishes easily while a ~1,800-word
            # drafting generation does not fit inside the deadline the caller
            # allowed. The API was reachable the whole time. The user was told
            # to check their network; the operator was told the same; and the
            # actual cause (a generation slower than the budget) appeared
            # nowhere. The wording now matches `error_kind`, and names the
            # budget that was actually exceeded.
            return self._error_response(
                f"The Gemini request did not finish within {effective_timeout:.0f}s. The service may be "
                "reachable but slower than this request's time budget -- try again, or allow more time "
                "for long generations.",
                error_kind="timeout",
            )

    async def _chat_with_retries(
        self, messages: list[ChatMessage], temperature: float, budget_seconds: float | None = None
    ) -> LLMResponse:
        payload = self._build_payload(messages, temperature)
        url = f"{GEMINI_API_BASE}/models/{self.model}:generateContent"
        client = self._get_client()
        # Transient failures (network blips, rate limiting) were previously
        # surfaced to the user on the very first failure with no retry -- in
        # practice this produced a visible ~20% failure rate on otherwise-
        # healthy requests during a regression run. A short, capped retry
        # with backoff absorbs those without materially hurting latency on
        # the (rare) genuinely-down case, since it's bounded by both
        # `max_retries` and the overall timeout wrapping this whole method.
        last_error_response: LLMResponse | None = None
        # Retries share the ONE budget rather than each restarting a fresh
        # one: a retry that cannot finish inside what's left is a retry that
        # only guarantees the caller waits longer for the same failure.
        attempt_deadline = None if budget_seconds is None else time.monotonic() + budget_seconds
        for attempt in range(self.max_retries + 1):
            attempt_started = time.monotonic()
            try:
                response = await client.post(url, params={"key": self.api_key}, json=payload)
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data, retry_count=attempt)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                log.warning(
                    "gemini_unavailable",
                    error_type=type(exc).__name__, error=str(exc), attempt=attempt,
                    elapsed_seconds=round(time.monotonic() - attempt_started, 2), model=self.model,
                )
                # `ConnectError` really is unreachable; `TimeoutException` is
                # not. Reporting both as unreachable is what sent an operator
                # looking at the network for a slow generation (see the
                # matching comment in `chat`).
                last_error_response = self._error_response(
                    "Gemini API is unreachable. Check your network connection and try again."
                    if isinstance(exc, httpx.ConnectError)
                    else "The Gemini request timed out before the model replied. The service may be "
                    "reachable but slower than the configured timeout.",
                    retry_count=attempt,
                    error_kind="timeout",
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                log.warning(
                    "gemini_http_error",
                    error_type=type(exc).__name__, status=status, attempt=attempt,
                    elapsed_seconds=round(time.monotonic() - attempt_started, 2), model=self.model,
                )
                last_error_response = self._error_response(
                    self._status_error_message(exc), retry_count=attempt,
                    error_kind=("rate_limit" if status == 429 else "http_error"),
                )
                if status not in _RETRIABLE_STATUS_CODES:
                    # 400/401/403/404 fail identically every time -- retrying
                    # them only spends budget that graceful degradation needs.
                    return last_error_response
            except Exception as exc:  # pragma: no cover  # noqa: BLE001 - last-resort provider fallback; any HTTP/SDK error must become a user-facing message, not a 500
                log.warning(
                    "gemini_request_failed", error_type=type(exc).__name__, error=str(exc), attempt=attempt,
                )
                return self._error_response(
                    "Gemini request failed unexpectedly. Please try again.",
                    retry_count=attempt, error_kind="provider_error",
                )
            if attempt >= self.max_retries:
                break
            backoff = self.retry_backoff_seconds[min(attempt, len(self.retry_backoff_seconds) - 1)]
            if attempt_deadline is not None:
                left = attempt_deadline - time.monotonic()
                if left <= backoff + MIN_VIABLE_CALL_SECONDS:
                    log.warning(
                        "gemini_retry_skipped_no_budget",
                        attempt=attempt, remaining_seconds=round(left, 2), model=self.model,
                    )
                    break
            await asyncio.sleep(backoff)
        # `last_error_response` is set by every `except` branch above, so it is
        # populated whenever the loop ran at all -- but the loop body does not
        # run when `max_retries` is negative, and returning `None` from a
        # method declared to return `LLMResponse` would surface downstream as
        # `AttributeError: 'NoneType' object has no attribute 'error'` in
        # whichever caller touched it first, far from the cause.
        return last_error_response or self._error_response(
            "Gemini request failed. Please try again.",
            retry_count=0,
            error_kind="provider_error",
        )

    async def stream(self, messages: list[ChatMessage], temperature: float = 0.1) -> AsyncIterator[str]:
        if not self.api_key:
            yield "Gemini API key is not configured."
            return
        if not has_time_for():
            log_exhausted("gemini_stream", provider=self.provider_name, model=self.model)
            return
        payload = self._build_payload(messages, temperature)
        url = f"{GEMINI_API_BASE}/models/{self.model}:streamGenerateContent"
        client = self._get_client()
        # `asyncio.wait_for` wraps a single coroutine, not an async
        # generator's ongoing iteration -- the overall ceiling here is
        # enforced with an explicit wall-clock deadline instead, checked
        # between chunks. Deliberately stops silently (not with an error
        # chunk) past the deadline: whatever text streamed so far is kept as
        # the partial answer, which is more useful than discarding it.
        effective_timeout = clamp_timeout(self.overall_timeout_seconds)
        stream_deadline = time.monotonic() + effective_timeout
        try:
            async with client.stream(
                "POST", url, params={"key": self.api_key, "alt": "sse"}, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if time.monotonic() > stream_deadline:
                        log.warning("gemini_stream_overall_timeout", timeout_seconds=effective_timeout)
                        return
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    chunk = json.loads(raw)
                    for candidate in chunk.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if part.get("thought"):
                                continue
                            text = part.get("text")
                            if text:
                                yield text
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("gemini_stream_unavailable", error=str(exc))
            yield "Gemini API is unreachable. Check your network connection and try again."
        except httpx.HTTPStatusError as exc:
            log.warning("gemini_stream_http_error", error=str(exc), status=exc.response.status_code)
            yield self._status_error_message(exc)
        except Exception as exc:  # pragma: no cover  # noqa: BLE001 - last-resort provider fallback for the streaming path
            log.warning("gemini_stream_failed", error=str(exc))
            yield "Gemini request failed unexpectedly. Please try again."

    def _health_timeout_seconds(self) -> float:
        """A short bound for the health probe, in seconds.

        `httpx.Timeout`'s `connect`/`read` are each `float | None` ("no
        timeout"). Summing them directly raised `TypeError` whenever either was
        unset -- and `health()` catches only transport errors, so that would
        have escaped as a 500 from `/health` rather than reporting the provider
        as down. `None` is treated as "no bound from that phase", leaving the
        5-second ceiling to do the work.
        """
        connect = self._timeout.connect or 0.0
        read = self._timeout.read or 0.0
        return min(5.0, connect + read) or 5.0

    async def health(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            response = await client.get(
                f"{GEMINI_API_BASE}/models/{self.model}",
                params={"key": self.api_key},
                timeout=self._health_timeout_seconds(),
            )
            response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            # `/health` reports the provider as down; it must not itself fail.
            # Narrow to transport/status errors so a bug in `_get_client()`
            # (bad config, missing key handling) surfaces instead of being
            # reported to operators as "Gemini is unreachable".
            log.warning("gemini_health_check_failed", error=str(exc), error_type=type(exc).__name__)
            return False
        return True

    async def search_grounded_urls(self, query: str, max_results: int = 5) -> list[str]:
        """Ask Gemini to answer using live Google Search grounding, and return
        the actual source URLs it grounded on -- never text the model itself
        wrote. A plain `chat()` call risks a hallucinated-but-plausible-looking
        .gov.in URL; every caller of this method still MUST fetch and verify
        whatever it gets back (this makes no claim about the URL being live,
        correct, or actually official) -- it only guarantees the URL came from
        a real search result, not from the model's own text generation.
        Returns an empty list on any failure (missing key, no grounding
        support, network/timeout) rather than raising -- callers already
        treat "no candidates" as a normal, expected outcome.
        """
        if not self.api_key:
            return []
        if not has_time_for():
            return []
        url = f"{GEMINI_API_BASE}/models/{self.model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": query}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.0},
        }
        client = self._get_client()
        try:
            response = await client.post(
                url, params={"key": self.api_key}, json=payload,
                timeout=clamp_timeout(min(30.0, self.overall_timeout_seconds)),
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, OSError) as exc:
            log.warning("gemini_search_grounding_failed", error=str(exc), error_type=type(exc).__name__)
            return []
        urls: list[str] = []
        for candidate in data.get("candidates") or []:
            chunks = (candidate.get("groundingMetadata") or {}).get("groundingChunks") or []
            for chunk in chunks:
                source_url = (chunk.get("web") or {}).get("uri")
                if source_url and source_url not in urls:
                    urls.append(source_url)
        return urls[:max_results]

    def _build_payload(self, messages: list[ChatMessage], temperature: float) -> dict[str, Any]:
        system_parts = [message.content for message in messages if message.role == "system"]
        contents = [
            {"role": "model" if message.role == "assistant" else "user", "parts": [{"text": message.content}]}
            for message in messages
            if message.role != "system"
        ]
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return payload

    def _parse_response(self, data: dict[str, Any], retry_count: int = 0) -> LLMResponse:
        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            # "Thinking" models (e.g. gemma-4-26b-a4b-it) return their internal
            # chain-of-thought as separate parts marked `"thought": true`,
            # interleaved before the real answer part(s) -- must be excluded or
            # the raw reasoning leaks into the user-facing answer.
            text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            content=text,
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            retry_count=retry_count,
        )

    def _status_error_message(self, exc: httpx.HTTPStatusError) -> str:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            return "Gemini API rate limit or quota reached. Please wait a moment and try again."
        if status in (401, 403):
            return "Gemini API key was rejected. Please check GEMINI_API_KEY."
        if status == 404:
            return f"Gemini model '{self.model}' was not found."
        return "Gemini API is currently unavailable. Please try again."

    def _error_response(self, error: str, retry_count: int = 0, error_kind: str = "provider_error") -> LLMResponse:
        return LLMResponse(
            content=error, model=self.model, provider=self.provider_name, prompt_tokens=0, completion_tokens=0,
            error=error, retry_count=retry_count, error_kind=error_kind,
        )
