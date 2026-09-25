from __future__ import annotations

import hashlib
import json
from pathlib import Path

import bson
import pytest

from scripts.verify_mongo_backup import verify_backup


def _backup(tmp_path: Path) -> Path:
    root = tmp_path / "backup"
    database = root / "legal_ai_assistant"
    database.mkdir(parents=True)
    payload = bson.BSON.encode({"_id": "one"}) + bson.BSON.encode({"_id": "two"})
    path = database / "documents.bson"
    path.write_bytes(payload)
    (root / "manifest.json").write_text(json.dumps({
        "database": "legal_ai_assistant",
        "collections": [{
            "name": "documents", "documents": 2, "file": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }), encoding="utf-8")
    return root


def test_backup_verifier_checks_hash_and_decoded_document_count(tmp_path: Path) -> None:
    assert verify_backup(_backup(tmp_path)) == {"collections": 1, "documents": 2}


def test_backup_verifier_rejects_tampering(tmp_path: Path) -> None:
    root = _backup(tmp_path)
    (root / "legal_ai_assistant" / "documents.bson").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum"):
        verify_backup(root)
