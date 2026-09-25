"""One-off verification script (2026-08-24): does `MongoVectorStore._atlas_vector_leg`
(app/rag/vector_store.py) actually work against a REAL Atlas Search $vectorSearch
deployment? No MongoDB Atlas cloud cluster exists in this environment, but MongoDB
ships `mongodb/mongodb-atlas-local` -- a Docker image bundling mongod + mongot that
provides genuine, offline Atlas Search / Vector Search. This script:

1. Connects to that container (started separately on localhost:27027).
2. Creates a REAL `vectorSearch`-type search index, same shape the app expects
   (`settings.mongodb_vector_index`, path "embedding", 1024 dims to match bge-m3).
3. Seeds a handful of synthetic chunks matching the app's real `embeddings_metadata`
   schema, including the exact Part 45/46 ownership metadata fields.
4. Calls the app's OWN `MongoVectorStore._atlas_vector_leg` and `.search()` directly
   (not reimplemented) against this index -- proving the code path for real.
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

ATLAS_LOCAL_URI = "mongodb://localhost:27027/?directConnection=true"
DB_NAME = "atlas_local_verification"
INDEX_NAME = settings.mongodb_vector_index
DIMS = 1024


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
    # Atlas requires the collection to already exist before a search index
    # can be created on it -- insert-then-delete a throwaway doc to force
    # creation (a real app's collection always exists by the time indexing
    # runs; this script starts from a brand-new, empty database).
    await collection.insert_one({"_id": "_bootstrap"})
    await collection.delete_one({"_id": "_bootstrap"})

    print("=== Step 1: create a real vectorSearch index (Atlas Search, via mongot) ===")
    index_def = {
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": DIMS, "similarity": "cosine"},
                {"type": "filter", "path": "metadata.owner_session_id"},
                {"type": "filter", "path": "metadata.owner_user_id"},
                {"type": "filter", "path": "metadata.act_name"},
                {"type": "filter", "path": "metadata.section_number"},
            ]
        },
    }
    try:
        result = await db.command({"createSearchIndexes": "embeddings_metadata", "indexes": [index_def]})
        print("createSearchIndexes result:", result)
    except Exception as exc:  # noqa: BLE001 - operator diagnostic script: reports whatever the server rejected the command with
        print("createSearchIndexes FAILED:", repr(exc))
        return

    print("\n=== Step 2: wait for the index to finish building (mongot async) ===")
    for attempt in range(30):
        indexes = await collection.list_search_indexes().to_list(None)
        status = next((idx.get("status") for idx in indexes if idx.get("name") == INDEX_NAME), None)
        print(f"  attempt {attempt+1}: status={status}")
        if status == "READY":
            break
        await asyncio.sleep(2)
    else:
        print("Index never became READY -- aborting.")
        return

    print("\n=== Step 3: seed synthetic chunks matching the real app schema ===")
    query_vec = _fake_vector(1)
    docs = [
        {
            "_id": "chunk-owned-by-user-1",
            "document_id": "doc-1",
            "text": "Section 2 of the Consumer Protection Act, 2019 defines key terms.",
            "embedding": query_vec,  # near-identical to the query -> should rank #1 when visible
            "metadata": {
                "source_document": "Consumer_Protection_Act.pdf",
                "act_name": "Consumer Protection Act, 2019",
                "section_number": "2",
                "owner_user_id": "user-1",
                "owner_session_id": None,
            },
            "namespace": "consumer_law",
        },
        {
            "_id": "chunk-owned-by-user-2",
            "document_id": "doc-2",
            "text": "Section 2 of a different Act, owned by a different user entirely.",
            "embedding": query_vec,  # also near-identical -- must NOT leak to user-1
            "metadata": {
                "source_document": "Other_Users_Private_Doc.pdf",
                "act_name": "Some Other Act",
                "section_number": "2",
                "owner_user_id": "user-2",
                "owner_session_id": None,
            },
            "namespace": "other",
        },
        {
            "_id": "chunk-global",
            "document_id": "doc-3",
            "text": "Section 5 of the Information Technology Act deals with digital signatures.",
            "embedding": _fake_vector(99),  # unrelated vector -> should rank low
            "metadata": {
                "source_document": "IT_Act.pdf",
                "act_name": "Information Technology Act",
                "section_number": "5",
                "owner_user_id": None,
                "owner_session_id": None,
            },
            "namespace": "cyber_law",
        },
    ]
    await collection.insert_many(docs)
    print(f"Inserted {len(docs)} synthetic chunks.")

    mongodb._client = client
    original_db_name = settings.mongodb_database
    settings.mongodb_database = DB_NAME
    settings.vector_search_backend = "atlas"
    try:
        print("\n=== Step 3b: wait for mongot to actually index the new documents ===")
        # A search index reporting READY means the INDEX DEFINITION is built
        # -- newly inserted documents still take a moment to replicate into
        # mongot's own Lucene index before a $vectorSearch query sees them.
        # Poll with a real $vectorSearch call (not just a status field) since
        # that's the only reliable signal for "are documents actually visible."
        store_probe = MongoVectorStore()
        for attempt in range(15):
            probe = await store_probe._atlas_vector_leg(query_vec, limit=5, filters={})
            print(f"  attempt {attempt+1}: {len(probe)} result(s) visible so far")
            if len(probe) >= len(docs):
                break
            await asyncio.sleep(2)

        print("\n=== Step 4: point the REAL MongoVectorStore at this deployment and query ===")
        store = MongoVectorStore()

        # 4a. Direct low-level call -- exactly what `search()` calls internally.
        raw_results = await store._atlas_vector_leg(query_vec, limit=5, filters={})
        print(f"\n_atlas_vector_leg (no filter) returned {len(raw_results)} results:")
        for r in raw_results:
            print(f"  {r.chunk_id}  score={r.score:.4f}  act={r.metadata.get('act_name')}")

        # 4b. Same Part 45/46 ownership `$or` filter shape ChatService actually builds,
        # confirming a real Atlas Search deployment honors it exactly like local scoring does.
        owner_filter = {
            "$or": [
                {"owner_session_id": [None, "session-x"], "owner_user_id": [None]},
                {"owner_user_id": ["user-1"]},
            ]
        }
        filtered_results = await store._atlas_vector_leg(query_vec, limit=5, filters=owner_filter)
        print(f"\n_atlas_vector_leg (user-1 ownership filter) returned {len(filtered_results)} results:")
        for r in filtered_results:
            print(f"  {r.chunk_id}  score={r.score:.4f}  owner={r.metadata.get('owner_user_id')}")

        visible_ids = {r.chunk_id for r in filtered_results}
        assert "chunk-owned-by-user-1" in visible_ids, "FAIL: user-1's own chunk should be visible"
        assert "chunk-global" in visible_ids, "FAIL: global (no-owner) chunk should be visible"
        assert "chunk-owned-by-user-2" not in visible_ids, "FAIL: user-2's private chunk LEAKED to user-1!"
        print("\nOwnership filter correctness: PASS (own chunk + global visible, other user's private chunk excluded)")

        # 4c. Full search() -- exercises the real fallback-on-error wiring too.
        full_results = await store.search(query_vec, "Section 2 Consumer Protection Act", top_k=5, filters={})
        print(f"\nsearch() (full hybrid path, vector leg only matters here) returned {len(full_results)} results")
        assert full_results, "FAIL: search() returned nothing"
        print("search() end-to-end: PASS")

        print("\n=== RESULT: real $vectorSearch against a live Atlas Search deployment WORKS ===")
    finally:
        settings.mongodb_database = original_db_name
        await client.drop_database(DB_NAME)
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
