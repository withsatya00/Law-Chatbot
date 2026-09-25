from app.core.config import settings
from app.llm.http_provider import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(self) -> None:
        super().__init__(
            provider_name="openai",
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
