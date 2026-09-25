"""Diagnostic repro (2026-09-24) for T026: why does turn 1
("mera landlord mera security deposit wapas nahi de raha hai") sometimes
reach `_finalize_rag_response`'s GK fallback via `not is_grounded` (a
citation/grounding-format problem) rather than `is_no_verified_context`
(an LLM Rule-2 refusal) -- the two have different fixes, and only the
second is what `_call_llm_with_grounding_retry` targets.

Matches ChatService._prepare_rag_context's exact call sequence, then builds
the real RAG messages and calls the real LLM directly, then runs the real
grounding validator, exactly mirroring answer()'s own logic.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.mongodb import mongodb
from app.intent.classifier import ConversationIntentMatch
from app.rag.citation import LegalCitationEngine as CitationEngine
from app.rag.reranker import LegalReranker
from app.rag.relevance import filter_relevant_context, reorder_by_relevance
from app.rag.retriever import LegalRetriever
from app.schemas.analysis import EntityResponse, IntentResponse
from app.services.chat_service import ChatService

QUERY = "mera landlord mera security deposit wapas nahi de raha hai"


async def main() -> None:
    await mongodb.connect()
    service = ChatService()
    retriever = LegalRetriever()
    reranker = LegalReranker()
    citation_engine = CitationEngine()

    filters = {"$or": [{"owner_session_id": [None], "owner_user_id": [None], "review_status": "approved"}]}
    rewritten, retrieved = await retriever.retrieve(QUERY, top_k=10, filters=filters, intent="LEGAL_QUESTION")
    ranked = await reranker.rerank(rewritten, retrieved, top_k=6, legal_category="Property Law")
    ranked = reorder_by_relevance(QUERY, ranked)
    ranked = filter_relevant_context(QUERY, rewritten, ranked, intent="LEGAL_QUESTION", language="hinglish")

    print(f"=== {len(ranked)} chunks survive the relevance gate ===")
    sources = []
    for chunk in ranked:
        print(f"  {chunk.score:.4f}  {chunk.metadata.get('source_document')}  act={chunk.metadata.get('act_name')!r}")
        citation = service._citation_from_chunk(chunk)
        sources.append(citation)
    print(f"\n=== sources built ({len(sources)}) ===")
    for s in sources:
        print(f"  act={s.act_name!r} doc={s.source_document!r} section={s.section!r}")

    intent = IntentResponse(intent="LEGAL_QUESTION", legal_category="Property Law", confidence=0.8, reason="repro")
    conv_intent = ConversationIntentMatch(intent="New Question", confidence=0.8, ambiguous=False, reason="repro")
    entities = EntityResponse(entities={}, confidence=0.5)
    messages = service._build_rag_messages(QUERY, "hinglish", intent, conv_intent, entities, ranked, "citizen")
    llm_response = await service.llm.chat(messages, temperature=0.1)
    answer = service._safe_llm_text(llm_response, "FALLBACK")
    print(f"\n=== LLM answer (attempt 1, temp=0.1) ===\n{answer[:800]}\n")

    from app.core.constants import is_no_verified_context
    print("is_no_verified_context:", is_no_verified_context(answer))
    is_grounded, reason = citation_engine.validate_grounding(answer, sources, ranked)
    print("is_grounded:", is_grounded, "reason:", reason)

    await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
