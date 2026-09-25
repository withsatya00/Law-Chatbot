from app.core.config import settings
from app.llm.http_provider import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    def __init__(self) -> None:
        super().__init__(
            provider_name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )
