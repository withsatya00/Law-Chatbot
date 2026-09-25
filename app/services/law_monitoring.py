"""Durable URL change detection and explicit, auditable KB publication.

No web content is fed to an LLM or automatically indexed. A changed official
page/PDF is evidence to review, not evidence that a law commenced or expired.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from bs4 import BeautifulSoup
from pymongo import ReturnDocument

from app.core.exceptions import BadRequestError, NotFoundError
from app.database.mongodb import mongodb
from app.rag.kb_jurisdiction import STATE_UT_CODES, document_metadata_fields, normalize_jurisdiction
from app.schemas.law_monitoring import (
    MonitorRequest,
    MonitorReviewRequest,
    official_url,
    ssl_context_for_host,
)

MONITORS = "law_source_monitors"
CHANGES = "law_source_changes"
MAX_BYTES = 4 * 1024 * 1024  # snapshots stored as BSON binary, below Mongo's document limit


def snapshot_fingerprint(body: bytes, content_type: str, selector: str | None = None) -> str:
    if selector:
        if content_type != "text/html":
            raise ValueError("Content selectors require HTML.")
        nodes = BeautifulSoup(body, "html.parser").select(selector)
        if not nodes or not any(node.get_text(strip=True) for node in nodes):
            raise ValueError("Legal content selector is missing or empty; source needs review.")
        body = "\n".join(str(node) for node in nodes).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


async def fetch_official_snapshot(url: str) -> tuple[bytes, str]:
    """Bounded fetch, no redirects/proxy inheritance/private-network destinations."""
    official_url(url)
    host = urlsplit(url).hostname
    addresses = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("Official source must resolve only to public addresses.")
    verify = ssl_context_for_host(host)
    async with httpx.AsyncClient(timeout=20, follow_redirects=False, trust_env=False, verify=verify) as client:  # noqa: SIM117 - stream depends on client
        async with client.stream("GET", url, headers={"User-Agent": "LegalKB-Monitor/1.0"}) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";")[0].lower()
            if content_type not in {
                "text/html", "text/plain", "application/pdf", "application/xml",
                "text/xml", "application/rss+xml", "application/atom+xml", "application/json",
            }:
                raise ValueError("Unsupported official-source response type.")
            body = bytearray()
            async for block in response.aiter_bytes():
                body.extend(block)
                if len(body) > MAX_BYTES:
                    raise ValueError("Official-source snapshot exceeds 4 MiB.")
            if not body:
                raise ValueError("Empty response is not a successful check.")
            return bytes(body), content_type


class LawMonitoringService:
    def __init__(self, db: Any = None, fetcher: Any = None, publisher: Any = None, invalidator: Any = None) -> None:
        self.db = db if db is not None else mongodb.db
        self.monitors = self.db[MONITORS]
        self.changes = self.db[CHANGES]
        self.fetcher = fetcher or fetch_official_snapshot
        self.publisher = publisher
        self.invalidator = invalidator

    async def invalidate(self) -> None:
        if self.invalidator is not None:
            await self.invalidator()
        else:
            from app.cache.response_cache import response_cache
            await response_cache.bump_generation(strict=True)

    async def ensure_indexes(self) -> None:
        await self.monitors.create_index("url", unique=True)
        await self.monitors.create_index([("enabled", 1), ("next_check_at", 1)])
        await self.changes.create_index([("status", 1), ("detected_at", -1)])

    async def register(self, request: MonitorRequest, actor: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        # URL identity is deterministic even before deployment provisions indexes.
        key = hashlib.sha256(request.url.encode()).hexdigest()
        await self.monitors.update_one(
            {"_id": key},
            {"$set": {**request.model_dump(), "updated_by": actor, "updated_at": now},
             "$setOnInsert": {"created_at": now, "next_check_at": now, "failures": 0}},
            upsert=True,
        )
        saved = await self.monitors.find_one({"_id": key})
        if saved is None:
            raise RuntimeError("Registered law monitor could not be reloaded.")
        return cast(dict[str, Any], saved)

    async def check_due(self, limit: int = 20) -> dict[str, int]:
        counts = {"checked": 0, "changed": 0, "failed": 0}
        now = datetime.now(UTC)
        cursor = self.monitors.find({"enabled": True, "next_check_at": {"$lte": now}}).sort("next_check_at", 1).limit(limit)
        async for monitor in cursor:
            result = await self.check(monitor["_id"])
            if result["status"] != "busy":
                counts["checked"] += 1
                counts["changed"] += int(result["status"] == "changed")
                counts["failed"] += int(result["status"] == "failed")
        return counts

    async def check(self, monitor_id: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        token = str(uuid4())
        monitor = await self.monitors.find_one_and_update(
            {"_id": monitor_id, "enabled": True, "$or": [
                {"lease_until": {"$exists": False}}, {"lease_until": {"$lte": now}},
            ]},
            {"$set": {"lease_until": now + timedelta(minutes=3), "lease_token": token}},
            return_document=ReturnDocument.AFTER,
        )
        if monitor is None:
            # Security/correctness finding G4: this lease-acquisition filter
            # (`enabled: True` + no live lease) fails to match for TWO very
            # different reasons -- a real monitor that's disabled or already
            # being checked by another worker (genuinely "busy"), and a
            # `monitor_id` that was never registered at all. Conflating them
            # reported 200 "busy" for a nonexistent id, which reads as
            # "try again shortly" when there is nothing to ever succeed.
            if await self.monitors.find_one({"_id": monitor_id}) is None:
                raise NotFoundError(f"Law monitor '{monitor_id}' not found.")
            return {"status": "busy"}
        owner = {"_id": monitor_id, "lease_token": token}
        try:
            # Total deadline, not just per-read HTTP timeout. The lease exceeds it.
            async with asyncio.timeout(60):
                body, content_type = await self.fetcher(monitor["url"])
            checksum = snapshot_fingerprint(body, content_type, monitor.get("content_selector"))
            changed = checksum != monitor.get("checksum")
            if changed:
                # Sequence keeps repeated A -> B -> A -> B changes reviewable;
                # retrying before advancing the baseline remains idempotent.
                change_id = hashlib.sha256(
                    f"{monitor_id}:{monitor.get('revision', 0)}:{checksum}".encode()
                ).hexdigest()
                await self.changes.update_one(
                    {"_id": change_id}, {"$setOnInsert": {
                        "monitor_id": monitor_id, "url": monitor["url"],
                        "state_code": monitor.get("state_code"), "topic": monitor["topic"],
                        "checksum": checksum, "previous_checksum": monitor.get("checksum"),
                        "snapshot": body, "content_type": content_type,
                        "snapshot_checksum": hashlib.sha256(body).hexdigest(),
                        "content_selector": monitor.get("content_selector"),
                        "detected_at": now, "status": "pending_review",
                        "kind": "baseline" if not monitor.get("checksum") else "content_change",
                    }}, upsert=True,
                )
            await self.monitors.update_one(owner, {"$set": {
                "checksum": checksum, "last_checked_at": now, "last_success_at": now,
                "revision": int(monitor.get("revision", 0)) + int(changed),
                "next_check_at": now + timedelta(hours=monitor["interval_hours"]),
                "failures": 0, "last_error": None,
            }, "$unset": {"lease_until": "", "lease_token": ""}})
            return {"status": "changed" if changed else "unchanged"}
        except Exception as exc:  # noqa: BLE001 - persist check failure and keep the worker available
            failures = int(monitor.get("failures", 0)) + 1
            await self.monitors.update_one(owner, {"$set": {
                "last_checked_at": now, "last_error": type(exc).__name__, "failures": failures,
                "next_check_at": now + timedelta(minutes=min(60 * 24, 5 * 2 ** min(failures - 1, 9))),
            }, "$unset": {"lease_until": "", "lease_token": ""}})
            return {"status": "failed", "error": type(exc).__name__}

    async def list_changes(self, status: str = "pending_review", limit: int = 100) -> list[dict[str, Any]]:
        return [item async for item in self.changes.find({"status": status}, {"snapshot": 0}).sort("detected_at", -1).limit(limit)]

    async def review(self, change_id: str, request: MonitorReviewRequest, actor: str) -> dict[str, Any]:
        from app.models.collections import UPLOADED_DOCUMENTS
        from app.services.kb_ingestion_service import KnowledgeBaseIngestionService

        change = await self.changes.find_one({"_id": change_id}, {"snapshot": 0})
        if change is None:
            raise NotFoundError("Monitored change not found.")
        if change["status"] not in {"pending_review", "publish_failed"}:
            raise BadRequestError("This change is already reviewed or being published.")
        if request.decision == "publish" and (not request.publications or not request.affected_versions_reviewed):
            raise BadRequestError("Publishing requires document metadata and confirmation that affected versions were reviewed.")
        if request.decision == "dismiss" and request.publications:
            raise BadRequestError("Dismissal cannot publish documents.")

        plans: list[tuple[str, dict[str, Any]]] = []
        previous_metadata: dict[str, Any] = {}
        seen: set[str] = set()
        for publication in request.publications:
            if publication.document_id in seen:
                raise BadRequestError("Duplicate document in publication plan.")
            seen.add(publication.document_id)
            document = await self.db[UPLOADED_DOCUMENTS].find_one({"_id": publication.document_id})
            if document is None:
                raise NotFoundError("Upload and index the reviewed document before publication.")
            metadata = document.get("metadata") or {}
            previous_metadata[publication.document_id] = metadata
            if any(document.get(key) or metadata.get(key) for key in ("owner_user_id", "owner_session_id")):
                raise BadRequestError("Private documents cannot be published by a monitor.")
            raw = {**publication.jurisdiction_metadata, "verification_status": "verified",
                   "verified_by": actor, "last_verified_at": datetime.now(UTC).isoformat(),
                   "metadata_provenance": "manual"}
            normalized = normalize_jurisdiction(raw, provenance="manual")
            fields = document_metadata_fields(normalized)
            if fields["review_status"] != "approved" or not fields.get("effective_from"):
                raise BadRequestError("Publication requires complete verified applicability and an effective-from date.")
            official_url(fields["source_url"])
            plans.append((publication.document_id, raw))

        if plans:
            # Refuse to begin publishing if cache freshness cannot be enforced.
            await self.invalidate()
        claim = await self.changes.update_one(
            {"_id": change_id, "status": change["status"]},
            {"$set": {"status": "publishing", "reviewed_by": actor,
                       "reviewed_at": datetime.now(UTC), "review_notes": request.notes,
                       "evidence_url": request.evidence_url, "publication_plan": [
                           {"document_id": doc_id, "metadata": raw} for doc_id, raw in plans
                       ], "previous_metadata": {**previous_metadata, **change.get("previous_metadata", {})}}},
        )
        if not claim.modified_count:
            raise BadRequestError("Another reviewer claimed this change.")
        publisher = self.publisher or KnowledgeBaseIngestionService()
        try:
            # Quarantine every affected document before changing any approved
            # version. A partial publication error re-quarantines the plan.
            for doc_id, raw in plans:
                await publisher.update_jurisdiction_metadata(doc_id, {**raw, "verification_status": "unverified"})
                await self.invalidate()
            for doc_id, raw in plans:
                result = await publisher.update_jurisdiction_metadata(doc_id, raw)
                if result.get("review_status") != "approved" or not result.get("chunks_updated"):
                    raise ValueError("Publication did not produce approved searchable chunks.")
                await self.invalidate()
        except Exception:
            quarantine_failures = []
            for doc_id, raw in plans:
                try:
                    await publisher.update_jurisdiction_metadata(doc_id, {**raw, "verification_status": "unverified"})
                    await self.invalidate()
                except Exception:  # noqa: BLE001 - report every document that could not be quarantined
                    quarantine_failures.append(doc_id)
            await self.changes.update_one({"_id": change_id}, {"$set": {
                "status": "publish_failed", "quarantine_failures": quarantine_failures,
            }})
            raise
        status = "published" if request.decision == "publish" else "dismissed"
        await self.changes.update_one({"_id": change_id}, {"$set": {
            "status": status, "completed_at": datetime.now(UTC), "document_ids": sorted(seen),
        }})
        return {"change_id": change_id, "status": status, "document_ids": sorted(seen)}

    async def coverage(self) -> dict[str, Any]:
        """Monitoring coverage only: never a claim all laws in a state are indexed."""
        now = datetime.now(UTC)
        rows = []
        async for monitor in self.monitors.find({}):
            last = monitor.get("last_success_at")
            if last and last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            pending = await self.changes.count_documents({
                "monitor_id": monitor["_id"], "status": {"$in": ["pending_review", "publishing", "publish_failed"]},
            })
            rows.append({
                **monitor, "pending_reviews": pending,
                "monitoring_status": "disabled" if not monitor["enabled"] else (
                    "never_checked" if last is None else "stale" if (
                        monitor.get("failures", 0) or now - last > timedelta(hours=monitor["interval_hours"] * 2)
                    ) else "review_pending" if pending else "checked"),
            })
        configured_states = {row.get("state_code") for row in rows if row["enabled"]}
        return {
            "scope": "configured URLs only; checked does not mean legally verified or complete",
            "monitors": rows,
            "unconfigured_state_codes": sorted(set(STATE_UT_CODES) - configured_states),
        }
