from app.llm.http_provider import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(
        self, api_key: str = "", model: str = "google/gemma-4-26b-a4b-it:free", base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        super().__init__(
            provider_name="openrouter",
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
