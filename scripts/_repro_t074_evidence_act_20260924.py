"""Faithful repro (2026-09-24) of T074 (QA_REPORT_100Q_FINAL_20260924.md
section 2.5/7.4): "admissibility of electronic evidence Bharatiya Sakshya
Adhiniyam" ranks a Delhi Police Academy FAQ document ahead of the real BSA
sections 63/64 text.

Matches ChatService._prepare_rag_context's exact call sequence (per
[[live-retrieval-repro-needs-reranker-and-cache-bump]]): retriever.retrieve()
-> reranker.rerank(top_k=6) -> reorder_by_relevance() -> filter_relevant_
context(), using the real production `LegalRetriever`/`LegalReranker`
directly, against the real live MongoDB. Read-only.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.mongodb import mongodb
from app.rag.reranker import LegalReranker
from app.rag.relevance import filter_relevant_context, reorder_by_relevance
from app.rag.retriever import LegalRetriever

QUERY = "admissibility of electronic evidence Bharatiya Sakshya Adhiniyam"


async def main() -> None:
    await mongodb.connect()
    retriever = LegalRetriever()
    reranker = LegalReranker()

    filters = {"$or": [{"owner_session_id": [None], "owner_user_id": [None], "review_status": "approved"}]}

    rewritten, retrieved = await retriever.retrieve(QUERY, top_k=10, filters=filters, intent="LEGAL_QUESTION")
    print(f"=== retrieve() -- {len(retrieved)} candidates, rewritten={rewritten!r} ===")
    for c in retrieved[:10]:
        print(f"  {c.score:.4f}  {c.metadata.get('source_document')}  act={c.metadata.get('act_name')!r} sec={c.metadata.get('section_number')!r} type={c.metadata.get('source_type')!r}")

    ranked = await reranker.rerank(rewritten, retrieved, top_k=6, legal_category="Evidence Law")
    print(f"\n=== rerank() -- top 6 ===")
    for c in ranked:
        print(f"  {c.score:.4f}  {c.metadata.get('source_document')}  act={c.metadata.get('act_name')!r} sec={c.metadata.get('section_number')!r} type={c.metadata.get('source_type')!r}")

    ranked = reorder_by_relevance(QUERY, ranked)
    final = filter_relevant_context(QUERY, rewritten, ranked, intent="LEGAL_QUESTION", language="english")
    print(f"\n=== after relevance gate -- {len(final)} survive ===")
    for c in final:
        print(f"  {c.score:.4f}  {c.metadata.get('source_document')}  act={c.metadata.get('act_name')!r} sec={c.metadata.get('section_number')!r}")

    await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
