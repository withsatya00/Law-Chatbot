import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from app.core.config import settings
from app.llm.base import ChatMessage, LLMProvider, LLMResponse

log = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout_seconds = float(timeout_seconds if timeout_seconds is not None else settings.llm_timeout)
        self.max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self.temperature = temperature if temperature is not None else settings.temperature
        self.top_p = top_p if top_p is not None else settings.top_p
        self.top_k = top_k if top_k is not None else settings.top_k
        self.retries = retries
        # Only ever set in tests, to inject an `httpx.MockTransport` -- `None`
        # in production means httpx uses its normal default transport.
        self._transport = transport

    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        payload = self._build_payload(messages, temperature, stream=False)
        start = self._now_ms()
        try:
            response = await self._post(payload)
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                return self._error_response(str(data["error"]), start)
            content = self._extract_content(data)
            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                provider=self.provider_name,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                latency_ms=self._elapsed_ms(start),
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("ollama_unavailable", error=str(exc))
            return self._error_response(self._timeout_or_unavailable_message(exc), start)
        except httpx.HTTPStatusError as exc:
            log.warning("ollama_http_error", error=str(exc))
            return self._error_response(self._status_error_message(exc), start)
        except Exception as exc:  # pragma: no cover  # noqa: BLE001 - last-resort provider fallback; a local Ollama can fail in ways httpx does not model
            log.warning("ollama_request_failed", error=str(exc))
            return self._error_response("Local LLM is unavailable. Please start Ollama.", start)

    async def stream(self, messages: list[ChatMessage], temperature: float = 0.1) -> AsyncIterator[str]:
        payload = self._build_payload(messages, temperature, stream=True)
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout_seconds, transport=self._transport) as client,
                client.stream("POST", self._api_url(), json=payload) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = self._extract_chunk(data)
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("ollama_stream_unavailable", error=str(exc))
            yield self._timeout_or_unavailable_message(exc)
        except httpx.HTTPStatusError as exc:
            log.warning("ollama_stream_http_error", error=str(exc))
            yield self._status_error_message(exc)
        except Exception as exc:  # pragma: no cover  # noqa: BLE001 - last-resort provider fallback for the streaming path
            log.warning("ollama_stream_failed", error=str(exc))
            yield "Local LLM is unavailable. Please start Ollama."

    async def health(self) -> bool:
        try:
            health_timeout = min(5.0, self.timeout_seconds)
            async with httpx.AsyncClient(timeout=health_timeout, transport=self._transport) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
            models = [model.get("name") for model in data.get("models", []) if isinstance(model, dict)]
        except (httpx.HTTPError, OSError, ValueError) as exc:
            # ValueError covers `response.json()` on a non-JSON body from
            # something that isn't Ollama listening on the configured port.
            log.warning("ollama_health_check_failed", error=str(exc), error_type=type(exc).__name__)
            return False
        return any(model == self.model for model in models)

    def _build_payload(self, messages: list[ChatMessage], temperature: float, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_ollama_message(message) for message in messages],
            "stream": stream,
            # Models with "thinking" capability (e.g. qwen3) emit a lengthy
            # chain-of-thought before the final answer unless this is off --
            # on constrained local hardware that pushes latency well past
            # LLM_TIMEOUT. Harmless no-op for models without the capability.
            "think": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_predict": self.max_tokens,
            },
        }
        return payload

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        # Only connection-establishment failures are worth retrying -- nothing
        # was sent yet, so a retry is always safe and cheap. A *read* timeout
        # (confirmed via real Ollama testing: it doesn't fail fast for a model
        # that isn't pulled -- it hangs until the request actually times out)
        # means the request WAS sent and the server is just slow/stuck;
        # retrying that just multiplies the wait for no benefit, so it's
        # deliberately excluded here and surfaces as a single-attempt failure
        # instead (same reasoning already applied to a 404 not being retried).
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self._transport) as client:
                    response = await client.post(self._api_url(), json=payload)
                    response.raise_for_status()
                    return response
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
                if attempt == self.retries - 1:
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama request failed")

    def _api_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _to_ollama_message(self, message: ChatMessage) -> dict[str, Any]:
        role = "user"
        if message.role == "assistant":
            role = "assistant"
        elif message.role == "system":
            role = "system"
        return {"role": role, "content": message.content}

    def _extract_content(self, data: dict[str, Any]) -> str:
        message = data.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content
        return ""

    _extract_chunk = _extract_content

    def _timeout_or_unavailable_message(self, exc: Exception) -> str:
        if isinstance(exc, httpx.ConnectError):
            return "Local LLM is unavailable. Please start Ollama."
        return (
            f"Ollama did not respond within {self.timeout_seconds:.0f}s. The model may still be "
            f"loading, or '{self.model}' may not be pulled -- run `ollama pull {self.model}` and try again."
        )

    def _status_error_message(self, exc: httpx.HTTPStatusError) -> str:
        if exc.response is not None and exc.response.status_code == 404:
            return "Requested model not found. Would you like to download it?"
        return "Local LLM is unavailable. Please start Ollama."

    def _error_response(self, error: str, started_at_ms: float) -> LLMResponse:
        return LLMResponse(
            content=error,
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=0,
            completion_tokens=0,
            error=error,
            latency_ms=self._elapsed_ms(started_at_ms),
        )

    def _now_ms(self) -> float:
        import time

        return time.perf_counter() * 1000

    def _elapsed_ms(self, started_at_ms: float) -> float:
        return max(0.0, self._now_ms() - started_at_ms)
