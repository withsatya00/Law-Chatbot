"""Idempotent Phase 3 metadata backfill; never overwrites existing values."""

import asyncio

from app.database.mongodb import mongodb
from app.models.collections import DOCUMENT_VERSIONS

DEFAULTS = {
    "source_version": "legacy-unversioned",
    "effective_date": None,
    "amendment_status": "unknown",
    "last_verified_date": None,
    "verification_status": "pending_review",
    "jurisdiction": "India",
}


async def migrate() -> dict[str, int]:
    collection = mongodb.db[DOCUMENT_VERSIONS]
    changed: dict[str, int] = {}
    for field, default in DEFAULTS.items():
        result = await collection.update_many({field: {"$exists": False}}, {"$set": {field: default}})
        changed[field] = result.modified_count
    return changed


async def main() -> None:
    await mongodb.connect()
    try:
        print(await migrate())
    finally:
        await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
