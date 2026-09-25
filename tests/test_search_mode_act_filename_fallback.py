"""Regression test for the `mode="act"` filename-fallback fix (QA session
2026-09-24, follow-up to BUG-109): `act_name` metadata is extracted from
document text at ingest time and is unreliable across this corpus (a
pre-existing, documented issue -- see `app/rag/citation.py`) -- most
Negotiable Instruments Act chunks carry `act_name=None` or a garbled
fragment, with only 1 of 47 carrying the clean literal string "Negotiable
Instruments Act". An exact-`act_name`-match-only lookup found that single
mislabeled-looking chunk and stopped there, missing the other 46 identified
just as reliably by their shared source filename.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from app.database.mongodb import mongodb
from app.services.search_service import SearchService


def _fake_collection(exact_match_docs, filename_match_docs):
    collection = MagicMock()

    def find(mongo_filter):
        cursor = MagicMock()
        if "$regex" in str(mongo_filter.get("metadata.source_document", "")):
            docs = filename_match_docs
        else:
            docs = exact_match_docs

        async def _iter():
            for doc in docs:
                yield doc

        cursor.limit.return_value = _iter()
        return cursor

    collection.find = MagicMock(side_effect=find)
    return collection


def test_falls_back_to_filename_match_when_exact_act_name_match_is_too_sparse():
    service = SearchService()
    service.retriever.vector_store._mongo_filter = MagicMock(side_effect=lambda f: dict(f))

    exact_match_docs = [
        {"_id": "c1", "text": "t1", "metadata": {"act_name": "Negotiable Instruments Act", "source_document": "ni_act.pdf"}},
    ]
    filename_match_docs = [
        {"_id": f"c{i}", "text": f"t{i}", "metadata": {"act_name": None, "source_document": "Negotiable_Instruments_Act_1881_Complete_Act.pdf"}}
        for i in range(2, 7)
    ]

    original_client = mongodb._client
    mongodb._client = {"legal_ai_assistant": {"embeddings_metadata": _fake_collection(exact_match_docs, filename_match_docs)}}
    try:
        async def run():
            return await service._find_by_act_name("Negotiable Instruments Act", {})

        results = asyncio.run(run())
    finally:
        mongodb._client = original_client

    assert len(results) == 5
    assert all(r.metadata["source_document"] == "Negotiable_Instruments_Act_1881_Complete_Act.pdf" for r in results)


def test_exact_match_used_directly_when_it_has_enough_chunks():
    service = SearchService()
    service.retriever.vector_store._mongo_filter = MagicMock(side_effect=lambda f: dict(f))

    exact_match_docs = [
        {"_id": f"c{i}", "text": f"t{i}", "metadata": {"act_name": "Right to Information Act", "source_document": "rti.pdf"}}
        for i in range(5)
    ]

    original_client = mongodb._client
    mongodb._client = {"legal_ai_assistant": {"embeddings_metadata": _fake_collection(exact_match_docs, [])}}
    try:
        async def run():
            return await service._find_by_act_name("Right to Information Act", {})

        results = asyncio.run(run())
    finally:
        mongodb._client = original_client

    assert len(results) == 5
    assert all(r.metadata["source_document"] == "rti.pdf" for r in results)
