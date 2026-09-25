from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import RetrievedChunk


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["semantic", "keyword", "hybrid", "metadata", "section", "act"] = "hybrid"
    top_k: int = Field(default=8, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)
    language: str | None = None


class SearchResponse(BaseModel):
    query: str
    rewritten_query: str
    results: list[RetrievedChunk]
    latency_ms: float
