from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Set only by providers reporting a known, user-facing failure mode (e.g.
    # "connection_failed", "model_not_found") -- lets callers distinguish real
    # generated content from a friendly error message surfaced through the
    # same `content` field, without every consumer needing its own detection.
    error: str | None = None
    # Machine-readable companion to `error`, so callers can branch on WHAT
    # went wrong without substring-matching a human sentence (which is what
    # `ResilientLLMProvider._is_retryable` and `app.llm.safety._LLM_ERROR_PHRASES`
    # both had to do). One of: "timeout", "rate_limit", "http_error",
    # "auth", "not_found", "deadline_exhausted", "provider_error",
    # "not_configured". Only meaningful when `error` is set.
    error_kind: str | None = None
    # Number of retry attempts a provider made before returning this response
    # (0 if it succeeded/failed on the first try). Optional/best-effort --
    # only `GeminiProvider` currently populates it (Part 29 performance
    # report requirement); every other provider defaults to 0, which is a
    # harmless under-report rather than a wrong one.
    retry_count: int = 0
    # Wall-clock time the provider spent on this call, when it measured one.
    # `OllamaProvider` has always passed this; until Phase 1 it was not a
    # declared field, so pydantic's default `extra="ignore"` dropped it
    # silently and every Ollama latency measurement was discarded on
    # construction. Optional because no other provider reports it.
    latency_ms: float | None = None


class LLMProvider(ABC):
    provider_name: str

    @abstractmethod
    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        raise NotImplementedError

    # Deliberately NOT `async def`. Every implementation of this method is an
    # async GENERATOR (`async def` + `yield`), so calling it returns an
    # `AsyncIterator[str]` directly and every call site consumes it as
    # `async for chunk in provider.stream(...)`. Declared `async def` here, the
    # signature instead promised `Coroutine[Any, Any, AsyncIterator[str]]` --
    # meaning an implementation written to match the base class literally
    # (a plain coroutine returning an iterator) would have broken every one of
    # those call sites, which would have had to `await` first. A plain `def`
    # returning `AsyncIterator[str]` is the accurate declaration and accepts
    # the async-generator implementations as valid overrides.
    @abstractmethod
    def stream(self, messages: list[ChatMessage], temperature: float = 0.1) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> bool:
        raise NotImplementedError
