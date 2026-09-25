"""JSONB compatibility repository used while Mongo stays dedicated to RAG."""

import json
from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID, uuid4

from app.database.postgresql import postgresql


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def dump_payload(document: dict[str, Any]) -> str:
    return json.dumps(document, default=_json_default, ensure_ascii=False)


# Every payload written through `dump_payload` runs `datetime`/`date` values
# through `_json_default` -> `.isoformat()` (there is no other way to put a
# datetime in a JSON document), so any reader downstream of `load_payload`
# that expects `created_at`/`updated_at` to still be `datetime` objects --
# `app/drafting/engine.py::history()` (fixed directly, "BUG-02"),
# `app/api/drafting.py::list_draft_versions` (same crash,
# `version["created_at"].isoformat()`, "BUG-102"), and
# `ConversationMemoryRepository.upsert_by_session` (silently failed every
# write after the first with `asyncpg`'s
# "expected a datetime.date or datetime.datetime instance, got 'str'",
# logged as `long_term_memory_write_failed` -- confirmed live under the
# BUG-101 concurrency test) -- gets a `str` instead and breaks. Three
# independent call sites hitting the exact same round-trip bug means the bug
# is in the round-trip itself, not any one caller: converting these two
# well-known timestamp fields back to `datetime` here, once, at the source,
# fixes the whole class rather than the next call site that happens to get
# written.
_DATETIME_PAYLOAD_KEYS = ("created_at", "updated_at")


def load_payload(value: Any) -> dict[str, Any]:
    document = json.loads(value) if isinstance(value, str) else dict(value)
    for key in _DATETIME_PAYLOAD_KEYS:
        raw = document.get(key)
        if isinstance(raw, str):
            try:
                document[key] = datetime.fromisoformat(raw)
            except ValueError:
                pass  # not actually an ISO timestamp -- leave it as-is rather than guess
    return document


def prepare_document(document: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC)
    result = dict(document)
    result.setdefault("_id", str(uuid4()))
    result.setdefault("created_at", now)
    result.setdefault("updated_at", now)
    return result


async def insert_payload(table: str, document: dict[str, Any]) -> str:
    row = prepare_document(document)
    async with postgresql.acquire() as connection:
        await connection.execute(
            f"INSERT INTO {table} (id, session_id, user_id, payload, created_at, updated_at) "
            "VALUES ($1, $2, $3, $4::jsonb, $5, $6)",
            str(row["_id"]), row.get("session_id"), row.get("user_id") or row.get("owner_user_id"),
            dump_payload(row), row["created_at"], row["updated_at"],
        )
    document.update(row)
    return str(row["_id"])


async def find_payload(table: str, item_id: str) -> dict[str, Any] | None:
    async with postgresql.acquire() as connection:
        value = await connection.fetchval(f"SELECT payload FROM {table} WHERE id = $1", item_id)
    return load_payload(value) if value is not None else None


async def update_payload(table: str, item_id: str, updates: dict[str, Any]) -> bool:
    current = await find_payload(table, item_id)
    if current is None:
        return False
    now = datetime.now(UTC)
    current.update(updates)
    current["updated_at"] = now
    async with postgresql.acquire() as connection:
        result = await connection.execute(
            f"UPDATE {table} SET session_id=$2, user_id=$3, payload=$4::jsonb, updated_at=$5 WHERE id=$1",
            item_id, current.get("session_id"), current.get("user_id") or current.get("owner_user_id"),
            dump_payload(current), now,
        )
    return bool(result == "UPDATE 1")


async def delete_where(table: str, clause: str, *args: Any) -> int:
    async with postgresql.acquire() as connection:
        result = await connection.execute(f"DELETE FROM {table} WHERE {clause}", *args)
    return int(result.rsplit(" ", 1)[-1])
