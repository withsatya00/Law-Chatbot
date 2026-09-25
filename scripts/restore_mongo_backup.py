"""Restores the `mongodump` snapshot in `storage/backups/mongo_backup/` into the
configured MongoDB instance, without needing the MongoDB Database Tools
(`mongorestore`) to be installed -- it decodes the .bson files with the `bson`
module that ships with pymongo.

Read-only with respect to the application: it only writes documents into the
database named by MONGODB_DATABASE, and it refuses to touch a collection that
already has documents unless --drop is passed.

Usage:
    python scripts/restore_mongo_backup.py --dry-run
    python scripts/restore_mongo_backup.py
    python scripts/restore_mongo_backup.py --drop        # replace existing data

Afterwards run `python scripts/create_indexes.py` to provision indexes.
"""

import argparse
from pathlib import Path

import bson
from pymongo import MongoClient

from app.core.config import settings

DEFAULT_DUMP_DIR = Path(__file__).resolve().parent.parent / "storage" / "backups" / "mongo_backup"
BATCH_SIZE = 500


def restore_collection(db, path: Path, *, drop: bool, dry_run: bool) -> tuple[str, int, str]:
    name = path.stem
    collection = db[name]
    existing = collection.count_documents({})

    with path.open("rb") as handle:
        documents = list(bson.decode_file_iter(handle))

    if not documents:
        return name, 0, f"empty dump file (collection has {existing})"
    if existing and not drop:
        return name, 0, f"SKIPPED -- collection already holds {existing} documents (use --drop to replace)"
    if dry_run:
        return name, len(documents), f"would insert {len(documents)} (existing {existing})"

    if drop and existing:
        collection.drop()
    for start in range(0, len(documents), BATCH_SIZE):
        collection.insert_many(documents[start : start + BATCH_SIZE], ordered=False)
    return name, len(documents), f"inserted {len(documents)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=None)
    parser.add_argument("--drop", action="store_true", help="drop non-empty target collections first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dump_dir = args.dump_dir or (DEFAULT_DUMP_DIR / settings.mongodb_database)
    if not dump_dir.is_dir():
        print(f"No dump directory at {dump_dir}")
        return 1

    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db = client[settings.mongodb_database]
    print(f"Target: {settings.mongodb_uri} -> database {settings.mongodb_database!r}")
    print(f"Source: {dump_dir}\n")

    total = 0
    for path in sorted(dump_dir.glob("*.bson")):
        name, count, note = restore_collection(db, path, drop=args.drop, dry_run=args.dry_run)
        total += count
        print(f"  {name:<24} {note}")

    print(f"\n{'Would insert' if args.dry_run else 'Inserted'} {total} documents total.")
    if not args.dry_run:
        print("Next: python scripts/create_indexes.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
