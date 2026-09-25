"""Create a checksummed BSON backup without modifying the source database."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import bson
from pymongo import MongoClient

from app.core.config import settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = args.output or settings.backup_dir / f"mongo-{stamp}"
    database_dir = root / settings.mongodb_database
    if root.exists():
        raise SystemExit(f"Refusing to overwrite existing backup: {root}")
    database_dir.mkdir(parents=True)

    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db = client[settings.mongodb_database]
    manifest = {"created_at": datetime.now(UTC).isoformat(),
                "database": settings.mongodb_database, "collections": []}
    for name in sorted(db.list_collection_names()):
        path = database_dir / f"{name}.bson"
        count = 0
        with path.open("wb") as handle:
            for document in db[name].find({}):
                handle.write(bson.BSON.encode(document))
                count += 1
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["collections"].append({
            "name": name, "documents": count, "file": path.name, "sha256": digest,
        })
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Backup created at {root} with {len(manifest['collections'])} collections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
