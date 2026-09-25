"""Backfill Constitution article metadata. Dry-run unless ``--apply`` is supplied."""

import argparse
import asyncio

from app.database.mongodb import mongodb
from app.models.collections import EMBEDDINGS_METADATA
from app.rag.bm25_index import bm25_index

_SIGNATURES = ("THE CONSTITUTION OF INDIA", "भारत का संविधान")


async def run(*, apply: bool) -> tuple[int, int, list[str]]:
    await mongodb.connect()
    collection = mongodb.db[EMBEDDINGS_METADATA]
    sources_by_signature: list[set[str]] = []
    for signature in _SIGNATURES:
        matching_sources: set[str] = set()
        async for item in collection.find(
            {"text": {"$regex": signature, "$options": "i"}}, {"metadata.source_document": 1}
        ):
            source = str((item.get("metadata") or {}).get("source_document") or "")
            if source:
                matching_sources.add(source)
        sources_by_signature.append(matching_sources)
    # A statute may merely mention "the Constitution of India".  The
    # official bilingual Constitution cover contains both exact titles; the
    # intersection prevents such references from reclassifying another Act.
    sources = set.intersection(*sources_by_signature) if sources_by_signature else set()
    query = {
        "metadata.source_document": {"$in": sorted(sources)},
        "metadata.section_number": {"$exists": True},
        "metadata.section_number_provenance": "heading",
    }
    candidates = await collection.count_documents(query)
    if not apply or not sources:
        return len(sources), candidates, sorted(sources)
    async for item in collection.find(query, {"metadata.section_number": 1}):
        number = str(item["metadata"]["section_number"])
        await collection.update_one(
            {"_id": item["_id"]},
            {
                "$set": {"metadata.instrument_type": "constitution", "metadata.article_number": number},
                "$unset": {"metadata.section_number": ""},
            },
        )
    await collection.update_many(
        {"metadata.source_document": {"$in": sorted(sources)}},
        {"$set": {"metadata.instrument_type": "constitution"}},
    )
    await bm25_index.rebuild()
    return len(sources), candidates, sorted(sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source_count, candidates, sources = asyncio.run(run(apply=args.apply))
    print({"apply": args.apply, "sources": source_count, "article_chunks": candidates, "source_names": sources})


if __name__ == "__main__":
    main()
