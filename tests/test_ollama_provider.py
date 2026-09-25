import asyncio
import json

import httpx

from app.llm.base import ChatMessage
from app.llm.ollama import OllamaProvider


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]


def _provider(handler, **kwargs) -> OllamaProvider:
    return OllamaProvider(transport=httpx.MockTransport(handler), **kwargs)


def test_chat_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["model"] == "qwen3:8b"
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "Hello there"},
                "done": True,
                "prompt_eval_count": 12,
                "eval_count": 5,
            },
        )

    # Explicit model= so this test's expectations don't silently depend on
    # whatever OLLAMA_MODEL happens to be set to in the local .env.
    response = asyncio.run(_provider(handler, model="qwen3:8b").chat(_messages()))
    assert response.content == "Hello there"
    assert response.error is None
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 5


def test_chat_connection_refused_returns_friendly_message() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        raise httpx.ConnectError("refused", request=request)

    response = asyncio.run(_provider(handler, retries=2).chat(_messages()))
    assert response.error
    assert "unavailable" in response.content.lower()
    assert call_count["n"] == 2  # confirms the retry path actually ran


def test_chat_model_not_found_is_not_retried() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})

    response = asyncio.run(_provider(handler, retries=3).chat(_messages()))
    assert response.error
    assert "not found" in response.content.lower()
    # A 404 means the request WAS answered -- retrying it is pointless and
    # should not happen, unlike a connection failure.
    assert call_count["n"] == 1


def test_chat_read_timeout_is_not_retried() -> None:
    # Regression: confirmed via real Ollama testing that it doesn't fail fast
    # for a model that isn't pulled -- the request is accepted but the
    # response never arrives until the request genuinely times out. Retrying
    # that (as opposed to a connection failure, where nothing was sent yet)
    # just multiplies the wait for no benefit -- worst case 3x the configured
    # timeout before ever surfacing the friendly error.
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        raise httpx.ReadTimeout("timed out", request=request)

    response = asyncio.run(_provider(handler, retries=3).chat(_messages()))
    assert response.error
    assert call_count["n"] == 1


def test_chat_unexpected_error_degrades_gracefully() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    response = asyncio.run(_provider(handler, retries=1).chat(_messages()))
    assert response.error
    assert response.content


def test_stream_success_concatenates_fragments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            json.dumps({"message": {"content": "Hel"}, "done": False}),
            json.dumps({"message": {"content": "lo"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        return httpx.Response(200, content=("\n".join(lines) + "\n").encode())

    async def run() -> list[str]:
        provider = _provider(handler)
        return [chunk async for chunk in provider.stream(_messages())]

    assert "".join(asyncio.run(run())) == "Hello"


def test_stream_connection_refused_yields_friendly_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async def run() -> list[str]:
        provider = _provider(handler)
        return [chunk async for chunk in provider.stream(_messages())]

    chunks = asyncio.run(run())
    assert len(chunks) == 1
    assert "unavailable" in chunks[0].lower()


def test_health_reachable_and_model_present() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}, {"name": "llama3.2:3b"}]})

    assert asyncio.run(_provider(handler, model="qwen3:8b").health()) is True


def test_health_reachable_but_model_absent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})

    assert asyncio.run(_provider(handler, model="qwen3:8b").health()) is False


def test_health_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert asyncio.run(_provider(handler).health()) is False
