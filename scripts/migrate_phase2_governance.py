"""Idempotent Phase 2 backfill for the legal-source registry.

Adds the governance fields Phase 2 introduced to records written before them.
Dry-run by default; `--apply` writes.

What it deliberately does NOT do:

  * It never promotes a record to `verified`. Verification asserts that a human
    compared the source against the issuing authority's own text, and no
    migration can perform that comparison. Records without evidence stay
    `unverified`, which is also why the schema default moved from
    `pending_review` to `unverified` -- a record nobody queued for review had
    not entered the review workflow at all.
  * It never invents `official_url`, `publication_date` or `last_verified_date`.
    A missing value stays missing.
  * It never deletes anything.

Usage:
    .venv\\Scripts\\python.exe scripts\\migrate_phase2_governance.py
    .venv\\Scripts\\python.exe scripts\\migrate_phase2_governance.py --apply
    .venv\\Scripts\\python.exe scripts\\migrate_phase2_governance.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.mongodb import mongodb
from app.models.collections import LEGAL_SOURCES

# field -> default applied only when the key is ABSENT. `None`/empty values that
# are already stored are left alone: an explicit empty is a recorded fact
# ("nobody supplied this"), not a gap for a migration to fill.
_DEFAULTS: dict[str, Any] = {
    "official_title": "",
    "source_type": "unknown",
    "issuing_authority": "",
    "publication_date": None,
    "checksum": "",
    "ingestion_version": "",
    "supersedes": [],
    "superseded_by": None,
    "chunk_ids": [],
    "reviewed_by": None,
    "reviewed_at": None,
    "review_notes": "",
    "evidence_url": "",
}


async def migrate(*, apply: bool) -> dict[str, Any]:
    await mongodb.connect()
    collection = mongodb.db[LEGAL_SOURCES]

    total = await collection.count_documents({})
    needing_defaults = 0
    downgraded_unverified = 0
    verified_without_evidence: list[str] = []

    async for record in collection.find({}):
        updates = {key: value for key, value in _DEFAULTS.items() if key not in record}

        # A pre-Phase-2 record could carry `verification_status="verified"` set
        # by the old `verify()`, which required no evidence at all. That is a
        # verification claim with nothing behind it, so it is reported and
        # returned to `pending_review` rather than silently kept.
        if record.get("verification_status") == "verified" and not (
            record.get("evidence_url") or updates.get("evidence_url")
        ):
            verified_without_evidence.append(str(record["_id"]))
            updates["verification_status"] = "pending_review"
            updates["review_notes"] = (
                "Phase 2 migration: verification predates the evidence requirement and "
                "must be re-reviewed against the issuing authority's own publication."
            )
            downgraded_unverified += 1

        if not updates:
            continue
        needing_defaults += 1
        if apply:
            await collection.update_one({"_id": record["_id"]}, {"$set": updates})

    return {
        "mode": "apply" if apply else "dry-run",
        "total_records": total,
        "records_updated" if apply else "records_needing_update": needing_defaults,
        "verified_without_evidence_returned_to_review": downgraded_unverified,
        "affected_source_ids": verified_without_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write changes (default is dry-run)")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args()

    report = asyncio.run(migrate(apply=args.apply))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
        if not args.apply:
            print("\nDry run. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
