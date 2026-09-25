"""Verification script (2026-09-23): does the REAL production Atlas Vector
Search index definition (`scripts.create_indexes.create_vector_search_index`)
actually let `MongoVectorStore.search()` (the exact method `LegalRetriever`
calls in production) succeed via `$vectorSearch`, or does it silently fall
back to `_local_cosine_leg` the way Phase 4A's "Atlas Vector Filter Repair"
already found for `owner_session_id`/`owner_user_id`/`section_number`?

Uses `mongodb/mongodb-atlas-local` (started separately on localhost:27027) --
a Docker image bundling mongod + mongot, a genuine offline Atlas Search /
Vector Search engine, not a guess about Atlas behavior. Calls the app's own
`create_vector_search_index()` and `MongoVectorStore` directly, never
reimplementing either.
"""

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings
from app.database.mongodb import mongodb
from app.rag.vector_store import MongoVectorStore
from scripts.create_indexes import create_vector_search_index

ATLAS_LOCAL_URI = "mongodb://localhost:27027/?directConnection=true"
DB_NAME = "atlas_readiness_verification_20260923"
DIMS = settings.embedding_dimensions


def _fake_vector(seed: int) -> list[float]:
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(DIMS)]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


async def main() -> None:
    client = AsyncIOMotorClient(ATLAS_LOCAL_URI)
    db = client[DB_NAME]
    collection = db["embeddings_metadata"]
    await collection.delete_many({})
    await collection.insert_one({"_id": "_bootstrap"})
    await collection.delete_one({"_id": "_bootstrap"})

    mongodb._client = client
    original_db_name = settings.mongodb_database
    settings.mongodb_database = DB_NAME
    settings.vector_search_backend = "atlas"

    try:
        print("=== Step 1: create the REAL production index definition (scripts/create_indexes.py) ===")
        await create_vector_search_index()

        print("\n=== Step 2: wait for the index to become READY ===")
        for attempt in range(30):
            indexes = await collection.list_search_indexes().to_list(None)
            status = next((idx.get("status") for idx in indexes if idx.get("name") == settings.mongodb_vector_index), None)
            print(f"  attempt {attempt+1}: status={status}")
            if status == "READY":
                break
            await asyncio.sleep(2)
        else:
            print("Index never became READY -- aborting.")
            return

        print("\n=== Step 3: seed one chunk exactly like a real ingested KB chunk (document_status=active) ===")
        query_vec = _fake_vector(1)
        await collection.insert_one({
            "_id": "chunk-1",
            "document_id": "doc-1",
            "text": "Section 2 of the Consumer Protection Act, 2019 defines key terms.",
            "embedding": query_vec,
            "metadata": {
                "source_document": "Consumer_Protection_Act.pdf",
                "act_name": "Consumer Protection Act, 2019",
                "section_number": "2",
                "owner_user_id": None,
                "owner_session_id": None,
                "document_status": "active",
            },
            "namespace": "consumer_law",
        })

        store = MongoVectorStore()
        print("\n=== Step 4: wait for mongot to index the new document (poll via a real $vectorSearch, no filter) ===")
        for attempt in range(15):
            probe = await store._atlas_vector_leg(query_vec, limit=5, filters={})
            print(f"  attempt {attempt+1}: {len(probe)} result(s) visible (no filter)")
            if probe:
                break
            await asyncio.sleep(2)

        print("\n=== Step 5: the EXACT filter shape ChatService/MongoVectorStore.search() actually sends ===")
        # `MongoVectorStore.search()` unconditionally injects
        # `document_status: "active"` via `_document_status_filter` on
        # EVERY call, production included -- this is what a live retrieval
        # request's filter dict looks like at minimum, before any
        # ownership/section filtering is even added.
        production_shaped_filters = {"document_status": "active"}

        print("\n--- 5a. Direct low-level _atlas_vector_leg call (raises on a real Atlas rejection) ---")
        try:
            direct_results = await store._atlas_vector_leg(query_vec, limit=5, filters=production_shaped_filters)
            print(f"_atlas_vector_leg SUCCEEDED: {len(direct_results)} result(s)")
            for r in direct_results:
                print(f"  {r.chunk_id}  score={r.score:.4f}  act={r.metadata.get('act_name')}")
        except Exception as exc:  # noqa: BLE001 - diagnostic script: report exactly what Atlas rejected the query with
            print(f"_atlas_vector_leg FAILED: {type(exc).__name__}: {exc}")

        print("\n--- 5b. Full search() (production entry point) -- does it silently fall back to local scan? ---")
        full_results = await store.search(query_vec, "Section 2 Consumer Protection Act", top_k=5, filters={})
        print(f"search() returned {len(full_results)} result(s) (see log line 'atlas_vector_search_unavailable' above if it fell back)")

        print("\n=== RESULT ===")
        print("If step 5a raised/logged a rejection above, the production Atlas index definition is")
        print("missing a filterable field for `document_status`, and EVERY real request would silently")
        print("fall back to the slow local scan in production -- Atlas would never actually be used.")
    finally:
        settings.mongodb_database = original_db_name
        await client.drop_database(DB_NAME)
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
