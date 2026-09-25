from collections.abc import AsyncIterator

import httpx
import structlog

from app.llm.base import ChatMessage, LLMProvider, LLMResponse

log = structlog.get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        if not self.api_key and self.provider_name != "ollama":
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider_name,
                prompt_tokens=0,
                completion_tokens=0,
            )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature,
            "stream": False,
        }
        # Mirrors the try/except -> friendly-error-message pattern already
        # used in gemini.py/ollama.py -- previously absent here, so any
        # failure (rate limit, bad key, network blip) propagated as a raw
        # exception instead of a graceful in-conversation message.
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("http_provider_unavailable", provider=self.provider_name, error=str(exc))
            return self._error_response(
                f"{self.provider_name} API is unreachable. Check your network connection and try again."
            )
        except httpx.HTTPStatusError as exc:
            log.warning("http_provider_http_error", provider=self.provider_name, error=str(exc))
            return self._error_response(self._status_error_message(exc))
        except Exception as exc:  # pragma: no cover  # noqa: BLE001 - last-resort provider fallback shared by every OpenAI-compatible backend
            log.warning("http_provider_request_failed", provider=self.provider_name, error=str(exc))
            return self._error_response(f"{self.provider_name} request failed unexpectedly. Please try again.")
        usage = data.get("usage", {})
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            provider=self.provider_name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def _status_error_message(self, exc: httpx.HTTPStatusError) -> str:
        status = exc.response.status_code if exc.response is not None else None
        if status == 401:
            return f"{self.provider_name} API key was rejected. Please check the configured API key."
        if status == 429:
            return f"{self.provider_name} API rate limit or quota reached. Please wait a moment and try again."
        return f"{self.provider_name} request failed unexpectedly. Please try again."

    def _error_response(self, error: str) -> LLMResponse:
        return LLMResponse(
            content=error, model=self.model, provider=self.provider_name,
            prompt_tokens=0, completion_tokens=0, error=error,
        )

    async def stream(self, messages: list[ChatMessage], temperature: float = 0.1) -> AsyncIterator[str]:
        response = await self.chat(messages, temperature)
        for token in response.content.split(" "):
            yield token + " "

    async def health(self) -> bool:
        return bool(self.api_key) or self.provider_name == "ollama"
