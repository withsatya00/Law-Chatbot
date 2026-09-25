"""Permanently purges `duplicate` and `failed` KB staging ledger rows (and,
where safe, their physical files) so the review queue starts clean.

Why "duplicate" and "failed" specifically: neither status is ever part of the
live, searchable Knowledge Base. `duplicate` means "this exact content was
already indexed under another record" (rejected at ingestion -- the content
itself lives on under the indexed record); `failed` means ingestion never
succeeded. Deleting them removes no answer chat can currently give; it only
tidies the audit ledger and reclaims disk space.

Safety rule, not optional: a `duplicate`/`failed` row's FILE is deleted only
if no `indexed` row currently points at that exact same physical path. A
handful do, here -- repeated re-uploads of byte-identical content left
several ledger rows sharing one surviving copy after
`repair_kb_staging_paths.py` repointed their stale paths. For those, the DB
ROW is still deleted (it is genuinely redundant ledger noise), but the file
is left untouched -- deleting it would take a live, currently-previewable
indexed document down with it.

A manifest of everything this removes -- full record bodies and file paths,
BEFORE deletion -- is written to `storage/kb_audit/` first, mirroring
`KnowledgeBaseIngestionService`'s own reconciliation-manifest convention, so
the delete decision is inspectable afterwards and never blind.

Read-only by default (dry run); pass `--apply` to actually delete.

Usage:
    python scripts/purge_kb_ledger_records.py             # dry run, prints + writes the plan
    python scripts/purge_kb_ledger_records.py --apply      # deletes DB rows and unshared files
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.mongodb import mongodb
from app.repositories.kb_staging import KnowledgeBaseStagingRepository

_PURGE_STATUSES = ("duplicate", "failed")
_MANIFEST_DIR = Path(__file__).resolve().parent.parent / "storage" / "kb_audit"


def _resolved_path(record: dict[str, Any]) -> str | None:
    raw = record.get("current_path") or record.get("staged_path") or record.get("archived_path")
    if not raw:
        return None
    path = Path(raw)
    return str(path.resolve()) if path.exists() else str(path)


async def plan_purge() -> dict[str, Any]:
    repository = KnowledgeBaseStagingRepository()

    protected_paths: set[str] = set()
    for record in await repository.find_by_status("indexed"):
        resolved = _resolved_path(record)
        if resolved:
            protected_paths.add(resolved)

    records = await repository.find_by_statuses(_PURGE_STATUSES)
    rows: list[dict[str, Any]] = []
    delete_file_paths: set[str] = set()
    for record in records:
        resolved = _resolved_path(record)
        shared_with_indexed = bool(resolved and resolved in protected_paths)
        rows.append(
            {
                "staging_id": str(record["_id"]),
                "original_filename": record.get("original_filename"),
                "status": record.get("status"),
                "content_hash": record.get("content_hash"),
                "reason": record.get("reason"),
                "recorded_path": record.get("current_path") or record.get("staged_path") or record.get("archived_path"),
                "resolved_path": resolved,
                "file_exists": bool(resolved and Path(resolved).is_file()),
                "shared_with_indexed": shared_with_indexed,
            }
        )
        if resolved and not shared_with_indexed and Path(resolved).is_file():
            delete_file_paths.add(resolved)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "records": rows,
        "record_count": len(rows),
        "files_to_delete": sorted(delete_file_paths),
        "files_to_delete_count": len(delete_file_paths),
        "records_with_protected_file": sum(1 for row in rows if row["shared_with_indexed"]),
    }


async def apply_purge(plan: dict[str, Any]) -> dict[str, int]:
    repository = KnowledgeBaseStagingRepository()
    deleted_records = 0
    for row in plan["records"]:
        if await repository.delete_by_id(row["staging_id"]):
            deleted_records += 1
    deleted_files = 0
    for path_str in plan["files_to_delete"]:
        path = Path(path_str)
        if path.is_file():
            path.unlink()
            deleted_files += 1
    return {"deleted_records": deleted_records, "deleted_files": deleted_files}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually delete. Default is a dry run.")
    args = parser.parse_args()

    await mongodb.connect()
    try:
        plan = await plan_purge()
        print(f"Duplicate/failed records: {plan['record_count']}")
        print(
            "  of which share a file with a live indexed record "
            f"(DB row only will be removed, file kept): {plan['records_with_protected_file']}"
        )
        print(f"Physical files eligible for deletion: {plan['files_to_delete_count']}")

        _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = _MANIFEST_DIR / f"purge_manifest_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        manifest_path.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
        print(f"Manifest written: {manifest_path}")

        if args.apply:
            result = await apply_purge(plan)
            print(f"\nDeleted {result['deleted_records']} ledger record(s) and {result['deleted_files']} file(s).")
        else:
            print("\nDry run only -- pass --apply to actually delete.")
    finally:
        await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
