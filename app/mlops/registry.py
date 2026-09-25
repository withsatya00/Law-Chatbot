from datetime import UTC, datetime
from typing import Any

from app.core.config import settings


class RuntimeVersionRegistry:
    def snapshot(self) -> dict[str, Any]:
        return {
            "model_provider": settings.llm_provider,
            "model_version": settings.model_version,
            "prompt_version": settings.prompt_version,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_version": settings.embedding_version,
            "vector_store_provider": settings.vector_store_provider,
            "vector_schema_version": settings.vector_schema_version,
            "captured_at": datetime.now(UTC).isoformat(),
        }

    def cache_namespace(self) -> str:
        return (
            f"{settings.llm_provider}:{settings.model_version}:{settings.prompt_version}:"
            f"{settings.embedding_model}:{settings.embedding_version}:{settings.vector_schema_version}"
        )
