"""Offline checksum and BSON-decode verification for a Mongo backup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import bson


def verify_backup(root: Path) -> dict[str, int]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    database_dir = root / manifest["database"]
    checked = documents = 0
    for record in manifest["collections"]:
        path = database_dir / record["file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch or missing backup file: {record['file']}")
        with path.open("rb") as handle:
            decoded = sum(1 for _ in bson.decode_file_iter(handle))
        if decoded != record["documents"]:
            raise ValueError(f"Document count mismatch: {record['file']}")
        checked += 1
        documents += decoded
    return {"collections": checked, "documents": documents}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    args = parser.parse_args()
    result = verify_backup(args.backup)
    print(f"Verified {result['collections']} collections and {result['documents']} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
