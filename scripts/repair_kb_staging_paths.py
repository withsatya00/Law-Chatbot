"""Repairs stale `current_path` pointers on terminal (`indexed`/`duplicate`)
KB staging ledger rows.

`current_path` records where a staged file physically lived at the time an
action last touched that row. Files under `storage/kb_staging` and
`storage/archive` have since drifted out of step with the ledger (repeated
test uploads of the same PDFs -- the `_2`/`_3` staging-collision suffixes
still visible in some `original_filename` values are the signature of this),
leaving a number of `indexed`/`duplicate` rows pointing at paths that no
longer exist, even though the exact same content is still present elsewhere
on disk. `POST /admin/knowledge-base/staging/{id}/file` (the KB review page's
PDF preview) is the ONLY thing this affects: `indexed` content is already
served for chat/search from `embeddings_metadata`/BM25, never by re-reading
the source file at query time, so this repair changes no retrieval behaviour.

Read-only by default (dry run); pass `--apply` to write. For each affected row
this looks for another file anywhere under `kb_staging_dir` / `archive_dir` /
`knowledge_base_dir` with the IDENTICAL `content_hash` and repoints
`current_path`/`staged_path` to it. A row with no matching file anywhere on
disk is left untouched and reported as unresolved -- never guessed at.

Usage:
    python scripts/repair_kb_staging_paths.py            # dry run, prints the plan
    python scripts/repair_kb_staging_paths.py --apply     # writes the repairs
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.database.mongodb import mongodb
from app.repositories.kb_staging import KnowledgeBaseStagingRepository

_REPAIR_STATUSES = ("indexed", "duplicate")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disk_hash_index() -> dict[str, Path]:
    """One file per hash is enough -- interchangeable copies of the same
    content, any of them is a valid repoint target."""
    index: dict[str, Path] = {}
    for directory in (settings.kb_staging_dir, settings.archive_dir, settings.knowledge_base_dir):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                index.setdefault(_hash_file(path), path)
    return index


async def plan_repairs() -> dict[str, list[dict[str, Any]]]:
    disk_index = _disk_hash_index()
    repository = KnowledgeBaseStagingRepository()
    records = await repository.find_by_statuses(_REPAIR_STATUSES)

    repairable: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for record in records:
        current = record.get("current_path") or record.get("staged_path") or record.get("archived_path")
        if current and Path(current).is_file():
            continue  # already correct
        content_hash = record.get("content_hash")
        found = disk_index.get(content_hash) if content_hash else None
        row = {
            "staging_id": str(record["_id"]),
            "original_filename": record.get("original_filename"),
            "status": record.get("status"),
            "recorded_path": current,
        }
        if found is not None:
            row["repointed_path"] = str(found)
            repairable.append(row)
        else:
            unresolved.append(row)
    return {"repairable": repairable, "unresolved": unresolved}


async def apply_repairs(repairable: list[dict[str, Any]]) -> int:
    repository = KnowledgeBaseStagingRepository()
    applied = 0
    for row in repairable:
        ok = await repository.update_by_id(
            row["staging_id"],
            {"current_path": row["repointed_path"], "staged_path": row["repointed_path"]},
        )
        applied += int(ok)
    return applied


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write the repairs. Default is a dry run.")
    args = parser.parse_args()

    await mongodb.connect()
    try:
        plan = await plan_repairs()
        print(f"Repairable (content found elsewhere on disk): {len(plan['repairable'])}")
        for row in plan["repairable"]:
            print(f"  [{row['status']}] {row['original_filename']} ({row['staging_id']})")
            print(f"      {row['recorded_path']!r} -> {row['repointed_path']!r}")
        print(f"\nUnresolved (no matching content on disk -- left untouched): {len(plan['unresolved'])}")
        for row in plan["unresolved"]:
            print(f"  [{row['status']}] {row['original_filename']} ({row['staging_id']}) recorded_path={row['recorded_path']!r}")

        if args.apply:
            applied = await apply_repairs(plan["repairable"])
            print(f"\nApplied {applied} repair(s).")
        else:
            print("\nDry run only -- pass --apply to write these repairs.")
    finally:
        await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
