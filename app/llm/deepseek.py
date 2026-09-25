from app.llm.http_provider import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    def __init__(
        self, api_key: str = "", model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com/v1",
    ) -> None:
        super().__init__(
            provider_name="deepseek",
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
