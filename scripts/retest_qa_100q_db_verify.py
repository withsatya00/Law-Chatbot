"""Direct-DB verification helpers for the 100-test QA run: confirms things
the HTTP API's own responses can't prove by themselves (e.g. that a DELETE
call actually removed rows, or that a chunk retrieved via /search really
exists in the underlying Mongo collection with the claimed metadata).
Appends findings into the shared results.json under the given tid.
"""
import asyncio
import json
import sys
from pathlib import Path

import asyncpg
import pymongo
import redis.asyncio as aioredis

OUT = Path("C:/Law Chatbot/Law Chatbot/qa-100q-retest-20260924")
POSTGRES_URL = "postgresql://legal_ai:legal_ai_local_dev_change_before_deploy@localhost:5432/legal_ai"
MONGO_URL = "mongodb://localhost:27017"
REDIS_URL = "redis://localhost:6379/0"


def load_results():
    return json.load(open(OUT / "results.json", encoding="utf-8"))


def save_results(results):
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)


async def verify_draft_postgres(tid, draft_id):
    results = load_results()
    entry = {"tid": tid, "meta": {"feature": "postgres_persistence_draft"}}
    try:
        conn = await asyncpg.connect(POSTGRES_URL)
        row = await conn.fetchrow("SELECT id, session_id, user_id, created_at FROM ai_legal_drafts WHERE id=$1", draft_id)
        entry["found"] = row is not None
        entry["row"] = dict(row) if row else None
        await conn.close()
    except Exception as exc:
        entry["error"] = str(exc)
    results[tid] = entry
    save_results(results)
    print(tid, "found:", entry.get("found"), entry.get("error", ""))


async def verify_mongo_chunk(tid, act_name_regex, section_number=None):
    results = load_results()
    entry = {"tid": tid, "meta": {"feature": "mongo_vector_verification"}}
    try:
        client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        db = client["legal_ai_assistant"]
        coll = db["embeddings_metadata"]
        q = {"metadata.act_name": {"$regex": act_name_regex, "$options": "i"}}
        if section_number:
            q["metadata.section_number"] = section_number
        count = coll.count_documents(q)
        sample = coll.find_one(q, {"metadata.source_document": 1, "metadata.act_name": 1, "metadata.section_number": 1, "_id": 0})
        entry["chunk_count"] = count
        entry["sample"] = sample
        client.close()
    except Exception as exc:
        entry["error"] = str(exc)
    results[tid] = entry
    save_results(results)
    print(tid, "chunk_count:", entry.get("chunk_count"), entry.get("error", ""))


async def verify_redis_session(tid, session_id):
    results = load_results()
    entry = {"tid": tid, "meta": {"feature": "redis_session_memory"}}
    try:
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        keys = []
        async for key in r.scan_iter(match=f"*{session_id}*"):
            keys.append(key)
        entry["matching_keys"] = keys
        entry["key_found"] = len(keys) > 0
        if keys:
            ttl = await r.ttl(keys[0])
            entry["ttl_seconds"] = ttl
        await r.aclose()
    except Exception as exc:
        entry["error"] = str(exc)
    results[tid] = entry
    save_results(results)
    print(tid, "key_found:", entry.get("key_found"), entry.get("matching_keys"), entry.get("error", ""))


async def verify_deletion_postgres_session(tid, session_id):
    results = load_results()
    entry = {"tid": tid, "meta": {"feature": "privacy_deletion_verification"}}
    try:
        conn = await asyncpg.connect(POSTGRES_URL)
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE 'ai_%'"
        )
        counts = {}
        for row in tables:
            t = row["table_name"]
            try:
                cnt = await conn.fetchval(f"SELECT count(*) FROM {t} WHERE session_id=$1", session_id)
                if cnt:
                    counts[t] = cnt
            except Exception:
                pass
        entry["remaining_rows_by_table"] = counts
        entry["fully_deleted"] = len(counts) == 0
        await conn.close()
    except Exception as exc:
        entry["error"] = str(exc)
    results[tid] = entry
    save_results(results)
    print(tid, "fully_deleted:", entry.get("fully_deleted"), entry.get("remaining_rows_by_table"), entry.get("error", ""))


async def verify_deletion_mongo_document(tid, document_id):
    results = load_results()
    entry = {"tid": tid, "meta": {"feature": "privacy_deletion_verification"}}
    try:
        client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        db = client["legal_ai_assistant"]
        remaining = db["embeddings_metadata"].count_documents({"document_id": document_id})
        entry["remaining_chunks"] = remaining
        entry["fully_deleted"] = remaining == 0
        client.close()
    except Exception as exc:
        entry["error"] = str(exc)
    results[tid] = entry
    save_results(results)
    print(tid, "fully_deleted:", entry.get("fully_deleted"), entry.get("error", ""))


async def main():
    mode = sys.argv[1]
    if mode == "draft":
        await verify_draft_postgres(sys.argv[2], sys.argv[3])
    elif mode == "mongo_chunk":
        section = sys.argv[4] if len(sys.argv) > 4 else None
        await verify_mongo_chunk(sys.argv[2], sys.argv[3], section)
    elif mode == "redis":
        await verify_redis_session(sys.argv[2], sys.argv[3])
    elif mode == "del_session":
        await verify_deletion_postgres_session(sys.argv[2], sys.argv[3])
    elif mode == "del_doc":
        await verify_deletion_mongo_document(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    asyncio.run(main())
