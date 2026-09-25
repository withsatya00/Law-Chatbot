from typing import Literal

from pydantic import BaseModel, Field


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict[str, str]]


class SessionFactRequest(BaseModel):
    kind: Literal["confirmed", "assumption"]
    key: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_.-]+$")
    value: str = Field(min_length=1, max_length=2000)


class MissingDetailsRequest(BaseModel):
    details: list[str] = Field(max_length=50)


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str | None = None
    rating: int = Field(ge=1, le=5)
    comment: str | None = None
    category: Literal["helpful", "wrong_language", "wrong_law", "missing_source", "unsafe_draft"] | None = None
