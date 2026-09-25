"""Idempotent Phase 2 case-record backfill.

MongoDB is schema-flexible, so existing Phase 1 records remain readable even
without this script. Running it makes the new dashboard fields explicit and
does not overwrite any field already present.

Usage: python scripts/migrate_phase2.py
"""

import asyncio

from app.database.mongodb import mongodb
from app.models.collections import CASES

DEFAULTS = {
    "legal_category": "",
    "parties": [],
    "tasks": [],
    "timeline": [],
    "next_action": "",
    "reminders": [],
    "linked_draft_ids": [],
    "resolved_conflicts": {},
    "evidence": [],
}


async def migrate() -> dict[str, int]:
    collection = mongodb.db[CASES]
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
