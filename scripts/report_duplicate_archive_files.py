"""Dry-run report: how much of `storage/archive` (and optionally other KB
storage directories) is redundant byte-identical copies of the same content.

Written for the incident `_reject_duplicate` in `app/services/
kb_ingestion_service.py` is fixed for going forward (see that function's own
docstring): before the fix, a source repeatedly re-discovered by automation/
sync was archived as a "new" duplicate every single cycle, since duplicate
rejection always copied the bytes under a fresh, collision-safe filename with
no check for "have we already preserved this exact content". The fix stops
NEW accumulation; this script only REPORTS on what already accumulated --
it never deletes or moves anything.

Usage:
    python scripts/report_duplicate_archive_files.py [DIR ...]

Defaults to `storage/archive` alone if no directory is given.
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "directories", nargs="*", type=Path, default=[Path("storage/archive")],
        help="Directories to scan (default: storage/archive)",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for directory in args.directories:
        if not directory.exists():
            print(f"skipping {directory} (does not exist)")
            continue
        files.extend(p for p in directory.rglob("*") if p.is_file())

    print(f"Scanning {len(files)} files across {len(args.directories)} director{'y' if len(args.directories) == 1 else 'ies'}...")

    by_hash: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    total_bytes = 0
    skipped = 0
    for index, path in enumerate(files, 1):
        try:
            size = path.stat().st_size
            content_hash = _hash_file(path)
        except OSError:
            # A live background process (KB automation/sync) is running
            # concurrently on this machine and can move/replace a file
            # between the directory listing above and this read -- skip it
            # rather than crash a read-only report on someone else's
            # in-flight write.
            skipped += 1
            continue
        total_bytes += size
        by_hash[content_hash].append((path, size))
        if index % 500 == 0:
            print(f"  ...{index}/{len(files)}")
    if skipped:
        print(f"  (skipped {skipped} file(s) that changed/disappeared mid-scan -- a live process is writing to this tree)")

    duplicate_groups = {h: entries for h, entries in by_hash.items() if len(entries) > 1}
    redundant_bytes = sum(
        size * (len(entries) - 1)
        for entries in duplicate_groups.values()
        for _, size in [entries[0]]
    )
    redundant_file_count = sum(len(entries) - 1 for entries in duplicate_groups.values())

    print()
    print(f"Total files scanned:     {len(files)}")
    print(f"Total size:              {_human_bytes(total_bytes)}")
    print(f"Distinct content hashes: {len(by_hash)}")
    print(f"Duplicate-content groups: {len(duplicate_groups)}")
    print(f"Redundant files (all but one copy per group): {redundant_file_count}")
    print(f"Space reclaimable if de-duplicated: {_human_bytes(redundant_bytes)}")
    print()

    top_groups = sorted(duplicate_groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]
    print("Top 20 groups by copy count:")
    for content_hash, entries in top_groups:
        entries_sorted = sorted(entries, key=lambda e: e[0].stat().st_mtime)
        size = entries_sorted[0][1]
        print(f"  {len(entries_sorted):4d} copies x {_human_bytes(size):>10}  {entries_sorted[0][0].name}  (hash {content_hash[:12]}...)")
        for path, _ in entries_sorted[:3]:
            print(f"        e.g. {path}")
        if len(entries_sorted) > 3:
            print(f"        ... and {len(entries_sorted) - 3} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
