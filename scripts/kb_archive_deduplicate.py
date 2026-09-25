"""Reclaims disk space from `storage/archive`/`storage/kb_staging` accumulating
redundant, byte-identical copies of the same rejected-duplicate content.

Root cause this cleans up after (now fixed going forward, see
`app.services.kb_ingestion_service.KnowledgeBaseIngestionService.
_reject_duplicate`'s own docstring): a source repeatedly re-discovered by
automation/sync was archived as a "new" duplicate every single cycle, because
archiving a rejected duplicate always copied the bytes under a fresh,
collision-safe filename with no check for "have we already preserved this
exact content". A dry run on 2026-09-23 found 2545 redundant files (3.1 GB)
across 1268 duplicate-content groups.

Same dry-run/manifest/apply shape as `scripts/kb_exact_deduplicate.py`, with
two differences suited to this specific, lower-risk case:

* Target is `storage/archive` and `storage/kb_staging`, not the live,
  indexed Knowledge Base -- files here were already REJECTED as duplicates
  (archive) or are in-flight uploads (staging), so they carry no
  `embeddings_metadata`/`document_versions` references to check by
  construction. This script verifies that anyway (defense in depth, cheap)
  and refuses to touch anything that unexpectedly IS referenced.
* Never deletes. Every copy but one per content hash is MOVED into a single
  timestamped holding directory (`storage/archive/_pending_deletion/
  <stamp>/`), never removed by this script -- consistent with this
  codebase's existing rule throughout `kb_ingestion_service.py` ("duplicates
  are moved, never deleted"). Permanently deleting that holding directory,
  once reviewed, is a deliberate, separate, manual step for a human to take.

Usage:
    python scripts/kb_archive_deduplicate.py
    python scripts/kb_archive_deduplicate.py --apply-manifest storage/operations/kb_archive_dedup/kb_archive_dedupe_<stamp>.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.database.mongodb import mongodb
from app.models.collections import DOCUMENT_VERSIONS, EMBEDDINGS_METADATA

_TARGET_DIRS = ("archive", "kb_staging")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_roots() -> list[Path]:
    base = settings.archive_dir.resolve().parent
    return [(base / name).resolve() for name in _TARGET_DIRS if (base / name).exists()]


def _canonical(paths: list[Path]) -> Path:
    """Which copy of a duplicate-content group to KEEP -- the oldest by
    modification time (the one automation/review is most likely to already
    reference by name), ties broken by the shortest/plainest filename (an
    unsuffixed original over a `_N`-suffixed later copy)."""
    def key(path: Path) -> tuple[float, bool, int, str]:
        stem = path.stem
        numbered = stem.rsplit("_", 1)[-1].isdigit()
        return (path.stat().st_mtime, numbered, len(path.name), path.name.casefold())

    return min(paths, key=key)


async def build_manifest() -> dict[str, Any]:
    roots = _target_roots()
    files: list[Path] = []
    for root in roots:
        files.extend(p for p in root.rglob("*") if p.is_file() and "_pending_deletion" not in p.parts)

    groups: dict[str, list[Path]] = defaultdict(list)
    skipped = 0
    for path in files:
        try:
            groups[file_hash(path)].append(path)
        except OSError:
            skipped += 1  # changed/disappeared mid-scan; a live process may share this tree

    plans: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    reclaimable_bytes = 0
    for digest, paths in sorted(groups.items()):
        if len(paths) < 2:
            continue
        referenced = []
        for path in paths:
            chunks = await mongodb.db[EMBEDDINGS_METADATA].count_documents({"metadata.source_document": path.name})
            versions = await mongodb.db[DOCUMENT_VERSIONS].count_documents(
                {"source_document": path.name, "document_status": {"$ne": "deleted"}}
            )
            if chunks > 0 or versions > 0:
                referenced.append(path.name)
        if referenced:
            # Should never happen for archive/staging content by construction
            # (see module docstring) -- if it does, this group is left alone
            # rather than guessed at.
            conflicts.append({"sha256": digest, "files": [str(p) for p in paths], "referenced": referenced})
            continue
        keep = _canonical(paths)
        move = [p for p in paths if p != keep]
        size = keep.stat().st_size
        reclaimable_bytes += size * len(move)
        plans.append({
            "sha256": digest,
            "keep": str(keep),
            "move": [str(p) for p in move],
            "size_bytes": size,
        })

    created = datetime.now(UTC)
    return {
        "schema_version": 1,
        "created_at": created.isoformat(),
        "roots": [str(r) for r in roots],
        "total_files": len(files),
        "files_skipped_mid_scan": skipped,
        "duplicate_groups": len(plans),
        "planned_move_files": sum(len(p["move"]) for p in plans),
        "reclaimable_bytes": reclaimable_bytes,
        "conflict_groups": conflicts,
        "plans": plans,
        "status": "dry_run",
    }


def write_manifest(manifest: dict[str, Any]) -> Path:
    output_dir = settings.operations_output_dir / "kb_archive_dedup"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"kb_archive_dedupe_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


async def apply_manifest(manifest_path: Path, *, expected_files: int, expected_groups: int, expected_move: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (expected_files, expected_groups, expected_move)
    actual = (manifest.get("total_files"), manifest.get("duplicate_groups"), manifest.get("planned_move_files"))
    if actual != expected:
        raise RuntimeError(f"Manifest counts {actual} do not match required counts {expected}.")
    if manifest.get("status") != "dry_run":
        raise RuntimeError("Manifest has already been applied.")
    # Unlike `kb_exact_deduplicate.py`, `conflict_groups` here does not block
    # applying: `build_manifest` already EXCLUDES every conflicting group
    # from `plans` (see its own loop) rather than flagging the whole run
    # ambiguous, so `conflict_groups` is purely informational -- what got
    # deliberately left untouched, for a human to look at separately.

    current = await build_manifest()
    if [(p["sha256"], p["keep"], p["move"]) for p in current["plans"]] != [
        (p["sha256"], p["keep"], p["move"]) for p in manifest["plans"]
    ]:
        raise RuntimeError("The target directories changed after the dry run; generate a new manifest.")

    holding_root = (settings.archive_dir / "_pending_deletion" / manifest_path.stem).resolve()
    holding_root.mkdir(parents=True, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    try:
        for plan in manifest["plans"]:
            for name in plan["move"]:
                source = Path(name).resolve()
                if not source.is_file() or file_hash(source) != plan["sha256"]:
                    raise RuntimeError(f"Source changed since dry run: {source}")
                destination = holding_root / f"{source.parent.name}__{source.name}"
                if destination.exists():
                    raise RuntimeError(f"Destination collision: {destination}")
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        if holding_root.exists() and not any(holding_root.iterdir()):
            holding_root.rmdir()
        raise

    manifest["status"] = "applied"
    manifest["applied_at"] = datetime.now(UTC).isoformat()
    manifest["holding_root"] = str(holding_root)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"moved": len(moved), "holding_root": str(holding_root)}


async def run(args: argparse.Namespace) -> int:
    await mongodb.connect()
    if args.apply_manifest:
        result = await apply_manifest(
            Path(args.apply_manifest),
            expected_files=args.expect_files, expected_groups=args.expect_groups, expected_move=args.expect_move,
        )
        print(json.dumps(result, indent=2))
        return 0
    manifest = await build_manifest()
    path = write_manifest(manifest)
    summary = {k: v for k, v in manifest.items() if k != "plans"}
    print(json.dumps({**summary, "manifest_path": str(path)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply-manifest")
    parser.add_argument("--expect-files", type=int)
    parser.add_argument("--expect-groups", type=int)
    parser.add_argument("--expect-move", type=int)
    args = parser.parse_args()
    if args.apply_manifest and None in {args.expect_files, args.expect_groups, args.expect_move}:
        parser.error("apply requires --expect-files, --expect-groups and --expect-move")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
