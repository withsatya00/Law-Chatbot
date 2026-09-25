import asyncio
import json

import httpx

from app.llm.base import ChatMessage
from app.llm.gemini import GeminiProvider


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]


def _provider(handler, **kwargs) -> GeminiProvider:
    return GeminiProvider(api_key="test-key", transport=httpx.MockTransport(handler), **kwargs)


def test_chat_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gemini-2.0-flash:generateContent"
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "Hello there"}]}}],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5},
            },
        )

    response = asyncio.run(_provider(handler, model="gemini-2.0-flash").chat(_messages()))
    assert response.content == "Hello there"
    assert response.error is None
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 5


def test_chat_excludes_thought_parts_from_answer() -> None:
    # Regression: "thinking" models (e.g. gemma-4-26b-a4b-it) return internal
    # chain-of-thought as separate parts marked `"thought": true`, interleaved
    # before the real answer -- a naive concatenation of all parts leaked that
    # raw reasoning straight into the user-facing answer.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Let me think about this step by step...", "thought": True},
                                {"text": "The final answer is 42."},
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 50},
            },
        )

    response = asyncio.run(_provider(handler).chat(_messages()))
    assert response.content == "The final answer is 42."
    assert "think about this" not in response.content


def test_chat_missing_api_key_returns_friendly_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not make a request without an API key")

    provider = GeminiProvider(api_key="", transport=httpx.MockTransport(handler))
    response = asyncio.run(provider.chat(_messages()))
    assert response.error
    assert "not configured" in response.content.lower()


def test_chat_rate_limited_returns_friendly_error_not_a_crash() -> None:
    # Regression: a real Gemini 429 previously propagated as an unhandled
    # httpx.HTTPStatusError all the way out of ChatService.answer(), crashing
    # the whole /chat request with a raw 500 instead of degrading gracefully
    # the way OllamaProvider already does.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "quota exceeded"}})

    response = asyncio.run(_provider(handler).chat(_messages()))
    assert response.error
    assert "rate limit" in response.content.lower() or "quota" in response.content.lower()


def test_chat_invalid_key_returns_friendly_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "API key not valid"}})

    response = asyncio.run(_provider(handler).chat(_messages()))
    assert response.error
    assert "key" in response.content.lower()


def test_chat_connection_error_returns_friendly_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    response = asyncio.run(_provider(handler).chat(_messages()))
    assert response.error
    assert "unreachable" in response.content.lower()


def test_stream_success_concatenates_fragments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}),
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "lo"}]}}]}),
        ]
        return httpx.Response(200, content=("\n".join(lines) + "\n").encode())

    async def run() -> list[str]:
        provider = _provider(handler)
        return [chunk async for chunk in provider.stream(_messages())]

    assert "".join(asyncio.run(run())) == "Hello"


def test_stream_excludes_thought_parts_from_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}]}}]}),
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "real"}]}}]}),
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": " answer"}]}}]}),
        ]
        return httpx.Response(200, content=("\n".join(lines) + "\n").encode())

    async def run() -> list[str]:
        provider = _provider(handler)
        return [chunk async for chunk in provider.stream(_messages())]

    assert "".join(asyncio.run(run())) == "real answer"


def test_stream_rate_limited_yields_friendly_message_not_a_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "quota exceeded"}})

    async def run() -> list[str]:
        provider = _provider(handler)
        return [chunk async for chunk in provider.stream(_messages())]

    chunks = asyncio.run(run())
    assert len(chunks) == 1
    assert "rate limit" in chunks[0].lower() or "quota" in chunks[0].lower()


def test_health_true_when_model_reachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gemini-2.0-flash"
        return httpx.Response(200, json={"name": "models/gemini-2.0-flash"})

    assert asyncio.run(_provider(handler, model="gemini-2.0-flash").health()) is True


def test_health_false_when_rate_limited() -> None:
    # Regression: health() previously only checked `bool(api_key)`, so a
    # rate-limited or invalid key would still report "ok".
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "quota exceeded"}})

    assert asyncio.run(_provider(handler).health()) is False


def test_health_false_without_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not make a request without an API key")

    provider = GeminiProvider(api_key="", transport=httpx.MockTransport(handler))
    assert asyncio.run(provider.health()) is False


def test_chat_does_not_retry_generic_server_error() -> None:
    # Part 29: 500/502 were previously retried; narrowed to only 429/503/504
    # since a generic server error is just as likely to be a genuinely
    # broken request as a transient blip, and retrying it wasted latency
    # budget for little payoff.
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, json={"error": {"message": "server error"}})

    response = asyncio.run(_provider(handler, max_retries=3, retry_backoff_seconds=(0.01, 0.01, 0.01)).chat(_messages()))
    assert response.error
    assert len(calls) == 1


def test_chat_retries_503_up_to_max_retries() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    provider = _provider(handler, max_retries=2, retry_backoff_seconds=(0.01, 0.01))
    response = asyncio.run(provider.chat(_messages()))
    assert response.error
    assert len(calls) == 3


def test_chat_respects_overall_timeout_ceiling() -> None:
    # A hard wall-clock ceiling on the ENTIRE call (Part 29) -- independent
    # of httpx's own connect/read timeout machinery, which `MockTransport`
    # bypasses entirely, so this specifically exercises the `asyncio.wait_for`
    # wrapper around `_chat_with_retries`.
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "too slow"}]}}]})

    provider = GeminiProvider(
        api_key="test-key", transport=httpx.MockTransport(handler), overall_timeout_seconds=0.05, max_retries=0
    )
    response = asyncio.run(provider.chat(_messages()))
    assert response.error
    assert response.error_kind == "timeout"
    # Post-Phase-3 hardening (Phase 2, milestone C): this used to assert
    # "unreachable". The API is reachable in this test -- the handler answers,
    # just too slowly -- so that wording was wrong, and it was wrong in
    # production too: a slow drafting generation on a healthy provider told
    # the user to check their network. The message must name the timeout.
    assert "did not finish" in response.content.lower()
    assert "network" not in response.content.lower()


def test_a_slow_provider_and_an_unreachable_one_are_reported_differently() -> None:
    """Both set `error_kind="timeout"`; only one is a network problem, and an
    operator reading either message has to be able to tell which."""

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "late"}]}}]})

    unreachable = asyncio.run(_provider(refused).chat(_messages()))
    timed_out = asyncio.run(
        GeminiProvider(
            api_key="test-key",
            transport=httpx.MockTransport(slow),
            overall_timeout_seconds=0.05,
            max_retries=0,
        ).chat(_messages())
    )
    assert "unreachable" in unreachable.content.lower()
    assert "unreachable" not in timed_out.content.lower()


def test_get_client_reused_across_calls_for_same_instance() -> None:
    # The whole point of Part 29's connection-reuse fix -- a fresh
    # `httpx.AsyncClient` per call defeats pooling/keep-alive entirely.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    provider = _provider(handler)
    assert provider._get_client() is provider._get_client()


def test_stream_stops_at_overall_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}),
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": " more"}]}}]}),
        ]
        return httpx.Response(200, content=("\n".join(lines) + "\n").encode())

    async def run() -> list[str]:
        # A deadline already 1s in the past -- deterministically exercises
        # the "stop yielding once the deadline has passed" branch (a 0.0s
        # ceiling risked ties with `time.monotonic()`'s clock resolution)
        # rather than racing a real sleep against a real timeout.
        provider = GeminiProvider(
            api_key="test-key", transport=httpx.MockTransport(handler), overall_timeout_seconds=-1.0
        )
        return [chunk async for chunk in provider.stream(_messages())]

    assert asyncio.run(run()) == []
