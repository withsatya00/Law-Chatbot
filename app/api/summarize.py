import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.llm.base import ChatMessage
from app.llm.factory import LLMFactory

log = structlog.get_logger(__name__)

router = APIRouter(tags=["summarization"])


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=120_000)
    max_chars: int = Field(default=1200, ge=200, le=5000)


class SummarizeResponse(BaseModel):
    summary: str


def _truncated_fallback(text: str, max_chars: int) -> str:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(paragraphs)[:max_chars]


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    """Was previously a non-LLM stub (join non-blank lines + truncate) --
    see [[project_orphaned_prototype_cluster]]'s sibling memory. Distinct
    from the in-chat "Response Modification" summarize transform (see
    [[project_response_modification_engine]]), which needs a live
    conversation/session; this endpoint summarizes arbitrary standalone
    text with no session at all, so it gets its own minimal prompt rather
    than reusing that engine's conversation-shaped one.

    Falls back to the original truncation behavior (never a bare error) on
    any LLM failure, same graceful-degradation convention as
    `ChatService._safe_llm_text` elsewhere in this app.
    """
    try:
        llm = LLMFactory.create()
        response = await llm.chat(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Summarize the user's text faithfully and concisely, in "
                        f"at most {request.max_chars} characters. Do not add facts, "
                        "opinions, or legal citations not present in the original text."
                    ),
                ),
                ChatMessage(role="user", content=request.text),
            ]
        )
        if response.error or not response.content.strip():
            return SummarizeResponse(summary=_truncated_fallback(request.text, request.max_chars))
        return SummarizeResponse(summary=response.content.strip()[: request.max_chars])
    # defensive: this endpoint must never 500 over a provider hiccup
    except Exception as exc:  # noqa: BLE001 - summarization degrades to a truncated extract; a provider fault must not 500 this endpoint
        log.warning("summarize_llm_call_failed", error=str(exc))
        return SummarizeResponse(summary=_truncated_fallback(request.text, request.max_chars))
