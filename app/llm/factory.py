import structlog

from app.core.config import settings
from app.llm.base import LLMProvider
from app.llm.claude import ClaudeProvider
from app.llm.deepseek import DeepSeekProvider
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.ollama import OllamaProvider
from app.llm.openai import OpenAIProvider
from app.llm.openrouter import OpenRouterProvider
from app.llm.resilient import ResilientLLMProvider

log = structlog.get_logger(__name__)


class LLMFactory:
    @staticmethod
    def create(provider: str | None = None) -> LLMProvider:
        selected = (provider or settings.llm_provider).lower()
        if selected == "groq":
            return GroqProvider()
        if selected == "openai":
            return OpenAIProvider()
        if selected == "ollama":
            return OllamaProvider()
        if selected == "deepseek":
            return DeepSeekProvider(settings.deepseek_api_key, settings.deepseek_model, settings.deepseek_base_url)
        if selected == "openrouter":
            return OpenRouterProvider(settings.openrouter_api_key, settings.openrouter_model, settings.openrouter_base_url)
        if selected == "gemini":
            return GeminiProvider()
        if selected == "claude":
            return ClaudeProvider()
        raise ValueError(f"Unsupported LLM provider: {selected}")


    @staticmethod
    def create_resilient(provider: str | None = None) -> LLMProvider:
        """The primary provider wrapped in retry + ordered failover.

        Used where a silent provider blip degrades the PRODUCT rather than
        just one reply -- most importantly legal draft generation, where the
        fallback is a bare skeleton document the user may sign and file. See
        `app/llm/resilient.py` for the incident this exists to prevent.

        `LLM_FALLBACK_PROVIDERS` (comma-separated names) is optional; with it
        unset this is still strictly better than `create()`, because the
        primary is retried with backoff instead of failing on first blip.
        A fallback name that is unknown or that fails to construct (missing
        API key, unimportable SDK) is skipped with a warning rather than
        breaking startup for everyone else.
        """
        primary = LLMFactory.create(provider)
        primary_name = (provider or settings.llm_provider).lower()
        fallbacks: list[LLMProvider] = []
        for name in settings.llm_fallback_provider_list():
            if name == primary_name:
                continue
            try:
                fallbacks.append(LLMFactory.create(name))
            except Exception as exc:  # noqa: BLE001 - a bad fallback must never break the primary
                log.warning("llm_fallback_provider_unavailable", provider=name, error=str(exc))
        return ResilientLLMProvider(
            primary,
            fallbacks,
            max_attempts=settings.llm_max_retry_attempts,
            initial_backoff_seconds=settings.llm_retry_backoff_seconds,
        )
