"""Archive unreferenced exact duplicates from the shared Knowledge Base.

Dry-run is the default and writes a reviewable manifest. Applying requires
that exact manifest plus expected inventory counts. No file is deleted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.database.mongodb import mongodb
from app.models.collections import AUDIT_LOGS, DOCUMENT_VERSIONS, EMBEDDINGS_METADATA, USERS


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unreferenced_canonical(names: list[str]) -> str:
    """Prefer an unsuffixed/readable name when no DB source is authoritative."""
    def key(name: str) -> tuple[int, int, str]:
        stem = Path(name).stem
        numbered = bool(re.search(r"_\d+$", stem))
        return (int(numbered), len(name), name.casefold())

    return min(names, key=key)


async def build_manifest() -> dict[str, Any]:
    root = settings.knowledge_base_dir.resolve()
    files = sorted((path for path in root.iterdir() if path.is_file()), key=lambda p: p.name.casefold())
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[file_hash(path)].append(path)

    plans: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    duplicate_groups = 0
    archive_count = 0
    for digest, paths in sorted(groups.items()):
        if len(paths) < 2:
            continue
        duplicate_groups += 1
        references: dict[str, dict[str, int]] = {}
        for path in paths:
            references[path.name] = {
                "chunks": await mongodb.db[EMBEDDINGS_METADATA].count_documents(
                    {"metadata.source_document": path.name}
                ),
                "active_versions": await mongodb.db[DOCUMENT_VERSIONS].count_documents(
                    {"source_document": path.name, "document_status": {"$ne": "deleted"}}
                ),
            }
        referenced = [
            name for name, counts in references.items()
            if counts["chunks"] > 0 or counts["active_versions"] > 0
        ]
        if len(referenced) > 1:
            conflicts.append({"sha256": digest, "files": [p.name for p in paths], "references": references})
            continue
        keep = referenced[0] if referenced else unreferenced_canonical([p.name for p in paths])
        archive = [p.name for p in paths if p.name != keep]
        archive_count += len(archive)
        plans.append(
            {
                "sha256": digest,
                "keep": keep,
                "archive": archive,
                "references": references,
                "selection_reason": "sole_active_database_source" if referenced else "best_unreferenced_filename",
            }
        )

    created = datetime.now(UTC)
    return {
        "schema_version": 1,
        "created_at": created.isoformat(),
        "root": str(root),
        "total_files": len(files),
        "exact_duplicate_groups": duplicate_groups,
        "planned_archive_files": archive_count,
        "conflict_groups": conflicts,
        "plans": plans,
        "status": "dry_run",
    }


def write_manifest(manifest: dict[str, Any]) -> Path:
    output_dir = settings.operations_output_dir / "kb_deduplication"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"kb_exact_dedupe_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


async def apply_manifest(
    manifest_path: Path,
    *,
    expected_files: int,
    expected_groups: int,
    expected_archive: int,
    actor_email: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (expected_files, expected_groups, expected_archive)
    actual = (
        manifest.get("total_files"),
        manifest.get("exact_duplicate_groups"),
        manifest.get("planned_archive_files"),
    )
    if actual != expected:
        raise RuntimeError(f"Manifest counts {actual} do not match required counts {expected}.")
    if manifest.get("status") != "dry_run" or manifest.get("conflict_groups"):
        raise RuntimeError("Manifest is not an unapplied, conflict-free dry run.")

    root = settings.knowledge_base_dir.resolve()
    if Path(manifest["root"]).resolve() != root:
        raise RuntimeError("Manifest root does not match the configured Knowledge Base root.")
    current = await build_manifest()
    if [
        (item["sha256"], item["keep"], item["archive"])
        for item in current["plans"]
    ] != [
        (item["sha256"], item["keep"], item["archive"])
        for item in manifest["plans"]
    ]:
        raise RuntimeError("Knowledge Base changed after the dry run; generate a new manifest.")

    actor = await mongodb.db[USERS].find_one({"email": actor_email.casefold(), "role": {"$in": ["admin", "super_admin"]}})
    if actor is None:
        raise RuntimeError("actor-email must identify an existing administrator account.")

    archive_root = (settings.archive_dir / "kb_duplicates" / manifest_path.stem).resolve()
    archive_root.mkdir(parents=True, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    try:
        for plan in manifest["plans"]:
            for name in plan["archive"]:
                source = (root / name).resolve()
                destination = (archive_root / name).resolve()
                if source.parent != root or destination.parent != archive_root:
                    raise RuntimeError(f"Unsafe path in manifest: {name}")
                if not source.is_file() or file_hash(source) != plan["sha256"] or destination.exists():
                    raise RuntimeError(f"Source changed or destination exists: {name}")
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        if archive_root.exists() and not any(archive_root.iterdir()):
            archive_root.rmdir()
        raise

    manifest["status"] = "applied"
    manifest["applied_at"] = datetime.now(UTC).isoformat()
    manifest["archive_root"] = str(archive_root)
    manifest["actor_user_id"] = str(actor["_id"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    await mongodb.db[AUDIT_LOGS].insert_one(
        {
            "actor_user_id": str(actor["_id"]),
            "action": "kb_exact_duplicates_archived",
            "resource": "knowledge_base",
            "details": {
                "manifest": manifest_path.name,
                "archive_root": str(archive_root),
                "files_archived": len(moved),
                "files_deleted": 0,
            },
            "created_at": datetime.now(UTC),
        }
    )
    return {"archived": len(moved), "deleted": 0, "archive_root": str(archive_root)}


async def run(args: argparse.Namespace) -> int:
    await mongodb.connect()
    if args.apply_manifest:
        result = await apply_manifest(
            Path(args.apply_manifest),
            expected_files=args.expect_files,
            expected_groups=args.expect_groups,
            expected_archive=args.expect_archive,
            actor_email=args.actor_email,
        )
        print(json.dumps(result, indent=2))
        return 0
    manifest = await build_manifest()
    path = write_manifest(manifest)
    print(json.dumps({**manifest, "manifest_path": str(path)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-manifest")
    parser.add_argument("--expect-files", type=int)
    parser.add_argument("--expect-groups", type=int)
    parser.add_argument("--expect-archive", type=int)
    parser.add_argument("--actor-email")
    args = parser.parse_args()
    if args.apply_manifest and None in {
        args.expect_files, args.expect_groups, args.expect_archive, args.actor_email
    }:
        parser.error("apply requires all --expect-* values and --actor-email")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
