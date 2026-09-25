from collections.abc import AsyncIterator

import anthropic
import structlog
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from app.core.config import settings
from app.llm.base import ChatMessage, LLMProvider, LLMResponse

log = structlog.get_logger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider via the official ``anthropic`` SDK."""

    provider_name = "claude"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else settings.claude_api_key
        self.model = model or settings.claude_model
        self._client: AsyncAnthropic | None = None

    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        # `temperature` is intentionally not forwarded: Claude Opus 5 rejects sampling
        # parameters with a 400. Response variance is controlled via `effort` instead.
        if not self.api_key:
            return LLMResponse(content="", model=self.model, provider=self.provider_name, prompt_tokens=0, completion_tokens=0)
        system_prompt, turns = self._split_messages(messages)
        # Mirrors the try/except -> friendly-error-message pattern already used
        # in gemini.py/ollama.py -- previously absent here, so any Claude API
        # failure (rate limit, bad key, network blip) propagated as a raw
        # exception instead of a graceful in-conversation message.
        try:
            response = await self._client_instance().messages.create(
                model=self.model,
                max_tokens=settings.claude_max_tokens,
                system=system_prompt,
                messages=turns,
                output_config={"effort": "low"},
            )
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            log.warning("claude_unavailable", error=str(exc))
            return self._error_response("Claude API is unreachable. Check your network connection and try again.")
        except anthropic.APIStatusError as exc:
            log.warning("claude_http_error", error=str(exc), status=exc.status_code)
            return self._error_response(self._status_error_message(exc))
        except Exception as exc:  # pragma: no cover  # noqa: BLE001 - last-resort provider fallback; any SDK error must become a user-facing message, not a 500
            log.warning("claude_request_failed", error=str(exc))
            return self._error_response("Claude request failed unexpectedly. Please try again.")
        if response.stop_reason == "refusal":
            return LLMResponse(content="", model=self.model, provider=self.provider_name, prompt_tokens=0, completion_tokens=0)
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            content=text,
            model=response.model,
            provider=self.provider_name,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    async def stream(self, messages: list[ChatMessage], temperature: float = 0.1) -> AsyncIterator[str]:
        if not self.api_key:
            return
        system_prompt, turns = self._split_messages(messages)
        try:
            async with self._client_instance().messages.stream(
                model=self.model,
                max_tokens=settings.claude_max_tokens,
                system=system_prompt,
                messages=turns,
                output_config={"effort": "low"},
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            log.warning("claude_stream_unavailable", error=str(exc))
            yield "Claude API is unreachable. Check your network connection and try again."
        except anthropic.APIStatusError as exc:
            log.warning("claude_stream_http_error", error=str(exc), status=exc.status_code)
            yield self._status_error_message(exc)
        except Exception as exc:  # pragma: no cover  # noqa: BLE001 - last-resort provider fallback for the streaming path; the stream must yield a message, not raise mid-response
            log.warning("claude_stream_failed", error=str(exc))
            yield "Claude request failed unexpectedly. Please try again."

    def _status_error_message(self, exc: "anthropic.APIStatusError") -> str:
        if exc.status_code == 401:
            return "Claude API key was rejected. Please check CLAUDE_API_KEY."
        if exc.status_code == 429:
            return "Claude API rate limit or quota reached. Please wait a moment and try again."
        return "Claude request failed unexpectedly. Please try again."

    def _error_response(self, error: str) -> LLMResponse:
        return LLMResponse(
            content=error, model=self.model, provider=self.provider_name,
            prompt_tokens=0, completion_tokens=0, error=error,
        )

    async def health(self) -> bool:
        return bool(self.api_key)

    def _client_instance(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    def _split_messages(self, messages: list[ChatMessage]) -> tuple[str, list[MessageParam]]:
        """Splits our provider-neutral `ChatMessage` list into the system
        prompt plus the alternating turns the Anthropic API takes.

        `role` is narrowed to the two values the API accepts. `ChatMessage.
        role` is a free-form `str`, so anything else reaching here -- a
        "tool" or "function" role added for another provider -- would
        previously have been forwarded verbatim and rejected by the API
        with a 400 at request time. Anything that is not an assistant turn
        is sent as `user`, which is how the API models "content supplied by
        the caller".
        """
        system_prompt = "\n\n".join(message.content for message in messages if message.role == "system")
        turns: list[MessageParam] = [
            {
                "role": "assistant" if message.role == "assistant" else "user",
                "content": message.content,
            }
            for message in messages
            if message.role != "system"
        ]
        return system_prompt, turns
