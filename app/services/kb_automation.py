"""Persistent, fail-closed acquisition pipeline for official legal sources."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import httpx
import structlog
from fastapi import UploadFile
from pymongo import ReturnDocument
from starlette.datastructures import Headers

from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.database.mongodb import mongodb
from app.models.collections import (
    EMBEDDINGS_METADATA,
    KB_AUTOMATION_JOBS,
    KB_AUTOMATION_STATE,
    OPERATIONAL_EVENTS,
)
from app.rag.kb_jurisdiction import STATE_UT_CODES, document_metadata_fields, normalize_jurisdiction
from app.schemas.law_monitoring import official_url, ssl_context_for_host
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService
from app.utils.malware_scan import ClamAVScanner, MalwareScanner

log = structlog.get_logger(__name__)
_PDF = "application/pdf"
_HTML = "text/html"
_CHALLENGE_MARKERS = ("captcha", "access denied", "sign in to continue", "g-recaptcha", "h-captcha", "cf-challenge")


class ManualAccessRequiredError(ValueError):
    """The portal demands interactive human access (CAPTCHA, a login wall, or
    an explicit access block). Retrying this on a timer cannot succeed, so it
    is routed to a distinct terminal state rather than the retry/backoff path
    -- see `KnowledgeBaseAutomationService._process_claimed`."""


class PortalLayoutChangedError(ValueError):
    """A previously productive catalogue adapter fetched valid HTML but
    matched zero candidates. Silently returning an empty list here would make
    a redesigned portal indistinguishable from "nothing new was published" --
    see `OfficialCatalogueAdapter.discover`."""


def normalize_batch_codes(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    codes = tuple(dict.fromkeys(part.strip().upper() for part in raw.split(",") if part.strip()))
    unknown = set(codes) - {"IN", *STATE_UT_CODES}
    if unknown:
        raise ValueError(f"Unknown jurisdiction code(s): {', '.join(sorted(unknown))}")
    return codes


def _scope_query(codes: tuple[str, ...]) -> dict[str, Any]:
    if not codes:
        return {}
    return {"$or": [
        {"candidate.jurisdiction_code": {"$in": list(codes)}},
        {"candidate.applicable_state_codes": {"$in": list(codes)}},
    ]}


def _adapter_in_scope(adapter: Any, codes: tuple[str, ...]) -> bool:
    if not codes:
        return True
    config = getattr(adapter, "config", None)
    if config is None:  # manifests are filtered candidate-by-candidate
        return True
    bound = {config.jurisdiction_code, *config.applicable_state_codes}
    return bool(bound.intersection(codes))


def _candidate_in_scope(candidate: SourceCandidate, codes: tuple[str, ...]) -> bool:
    if not codes:
        return True
    return bool({candidate.jurisdiction_code, *candidate.applicable_state_codes}.intersection(codes))


@dataclass(frozen=True)
class SourceCandidate:
    adapter: str
    external_id: str
    title: str
    url: str
    jurisdiction_code: str
    document_type: str
    version: str
    filename: str
    act_number: str | None = None
    enactment_year: int | None = None
    effective_from: str | None = None
    language: str = "english"
    issuing_authority: str | None = None
    publication_date: str | None = None
    parent_external_id: str | None = None
    # Extra states/UTs a candidate binds in, beyond `jurisdiction_code` itself
    # -- a High Court routinely covers several States/UTs from one portal.
    applicable_state_codes: tuple[str, ...] = ()
    # Advisory only, from a deterministic keyword match on the surrounding
    # listing text. Never fed into `amends`/`supersedes` metadata -- that
    # linkage stays a human decision through `MonitorReviewRequest`.
    change_type: str = "unclassified"
    # Advisory review-queue ordering only -- see `app/rag/source_confidence.py`.
    # Never influences `verification_status`.
    confidence_score: float = 0.0

    @property
    def source_key(self) -> str:
        return f"{self.adapter}:{self.external_id}:{self.version}"

    @property
    def canonical_document_key(self) -> str:
        pieces = [self.jurisdiction_code, self.document_type, self.act_number or "na",
                  str(self.enactment_year or "na"), self.parent_external_id or self.title]
        slug = re.sub(r"[^a-z0-9]+", "-", "-".join(pieces).casefold()).strip("-")
        return slug[:120]


class SourceAdapter(Protocol):
    name: str

    async def discover(self, checkpoint: str | None) -> tuple[list[SourceCandidate], str | None]: ...


class ManifestSourceAdapter:
    """Adapter for a reviewed JSON manifest; useful as the first production source."""

    name = "official_manifest"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.kb_automation_manifest

    async def discover(self, checkpoint: str | None) -> tuple[list[SourceCandidate], str | None]:
        del checkpoint
        if not self.path.is_file():
            return [], None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        candidates = []
        for row in payload.get("sources", []):
            candidate = SourceCandidate(adapter=self.name, **row)
            official_url(candidate.url)
            candidates.append(candidate)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return candidates, digest


class RetrievalBenchmark(Protocol):
    async def evaluate(self, candidate: SourceCandidate, source_document: str) -> dict[str, Any]: ...


class CentralRetrievalBenchmark:
    """Exercise the real hybrid retriever after a document is approved."""

    def __init__(self) -> None:
        self._retriever: Any = None

    async def evaluate(self, candidate: SourceCandidate, source_document: str) -> dict[str, Any]:
        if self._retriever is None:
            from app.rag.retriever import LegalRetriever
            self._retriever = LegalRetriever()
        rewritten, results = await self._retriever.retrieve(
            candidate.title, top_k=5, filters={"source_document": source_document},
        )
        matched = [item for item in results if item.metadata.get("source_document") == source_document]
        return {
            "passed": bool(matched), "query": candidate.title, "rewritten_query": rewritten,
            "matches": len(matched), "top_score": matched[0].score if matched else 0.0,
        }


class OfficialDownloader:
    """Download with public-host validation at every redirect hop."""

    def __init__(self, max_bytes: int | None = None) -> None:
        self.max_bytes = max_bytes or settings.kb_automation_max_download_mb * 1024 * 1024

    async def fetch(self, url: str) -> tuple[bytes, str, str]:
        current = official_url(url)
        # Based on the ORIGINAL host, not re-evaluated per redirect hop: every
        # known-affected host (see `ssl_context_for_host`) serves directly,
        # with no redirect chain, so this is never wrong for them, and every
        # other host keeps the ordinary secure default regardless.
        verify = ssl_context_for_host(urlsplit(current).hostname)
        async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False, verify=verify) as client:
            for _ in range(4):
                # Reuse the monitor's DNS/SSRF checks for the current hop. It
                # also bounds snapshots, so larger PDFs use the stream below.
                await self._validate_public(current)
                async with client.stream("GET", current, headers={"User-Agent": "LegalKB-Acquirer/1.0"}) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Official source returned an empty redirect.")
                        current = official_url(urljoin(current, location))
                        continue
                    if response.status_code in {403, 429}:
                        raise ManualAccessRequiredError(
                            f"Official source returned HTTP {response.status_code}, "
                            "consistent with an interactive access block."
                        )
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";")[0].lower()
                    body = bytearray()
                    async for block in response.aiter_bytes():
                        body.extend(block)
                        if len(body) > self.max_bytes:
                            raise ValueError("Official document exceeds the configured download limit.")
                    data = bytes(body)
                    self._validate_payload(data, content_type)
                    return data, content_type, current
            raise ValueError("Official source exceeded the redirect limit.")

    @staticmethod
    async def _validate_public(url: str) -> None:
        # This call validates scheme/domain/DNS without trusting redirects.
        # A lightweight HEAD is intentionally avoided because many government
        # servers reject it; the shared helper's network fetch is replaced in
        # tests and the actual GET remains bounded below.
        official_url(url)
        import ipaddress
        import socket
        host = urlsplit(url).hostname
        addresses = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError("Official source must resolve only to public addresses.")

    @staticmethod
    def _validate_payload(data: bytes, content_type: str) -> None:
        if not data:
            raise ValueError("Official source returned an empty document.")
        if content_type == _PDF:
            if not data.startswith(b"%PDF-"):
                raise ValueError("PDF response does not contain a PDF signature.")
            if b"%%EOF" not in data[-4096:]:
                raise ValueError("Downloaded PDF is truncated or has no EOF marker.")
            return
        if content_type == _HTML:
            sample = data[:200_000].decode(errors="ignore").casefold()
            if "<html" not in sample and "<!doctype html" not in sample:
                raise ValueError("HTML response does not contain an HTML document signature.")
            if any(marker in sample for marker in _CHALLENGE_MARKERS):
                raise ManualAccessRequiredError(
                    "Official page requires interactive access and cannot be auto-ingested."
                )
            return
        raise ValueError("Automation accepts official PDF or HTML legal documents only.")


class KnowledgeBaseAutomationService:
    def __init__(
        self, db: Any = None, adapters: list[SourceAdapter] | None = None,
        downloader: Any = None, scanner: MalwareScanner | None = None, ingestion: Any = None,
        benchmark: RetrievalBenchmark | None = None,
    ) -> None:
        self.db = db if db is not None else mongodb.db
        self.jobs = self.db[KB_AUTOMATION_JOBS]
        self.state = self.db[KB_AUTOMATION_STATE]
        self.events = self.db[OPERATIONAL_EVENTS]
        if adapters is None:
            from app.services.kb_central_adapters import central_source_adapters
            from app.services.kb_high_court_adapters import high_court_source_adapters
            from app.services.kb_state_adapters import state_source_adapters
            registered: list[SourceAdapter] = [
                ManifestSourceAdapter(), *central_source_adapters(),
                *state_source_adapters(), *high_court_source_adapters(),
            ]
            allowed = {
                name.strip() for name in settings.kb_automation_adapter_allowlist.split(",")
                if name.strip()
            }
            known = {adapter.name for adapter in registered}
            unknown = allowed - known
            if unknown:
                raise ValueError(f"Unknown KB automation adapter(s): {', '.join(sorted(unknown))}")
            self.adapters = [adapter for adapter in registered if not allowed or adapter.name in allowed]
            self._load_registered_sources = True
        else:
            self.adapters = adapters
            self._load_registered_sources = False
        self.downloader = downloader or OfficialDownloader()
        self.scanner = scanner or ClamAVScanner()
        self.ingestion = ingestion or KnowledgeBaseIngestionService()
        self.benchmark = benchmark or CentralRetrievalBenchmark()

    async def ensure_indexes(self) -> None:
        await self.jobs.create_index("source_key", unique=True)
        await self.jobs.create_index([("status", 1), ("next_attempt_at", 1)])
        await self.jobs.create_index([("canonical_document_key", 1), ("created_at", -1)])
        await self.jobs.create_index([("status", 1), ("next_source_check_at", 1)])

    async def discover(self, jurisdiction_codes: tuple[str, ...] = ()) -> dict[str, int]:
        counts = {"discovered": 0, "already_known": 0, "adapter_failures": 0}
        adapters = list(self.adapters)
        if self._load_registered_sources:
            from app.models.collections import KB_SOURCE_ADAPTERS
            from app.services.kb_source_onboarding import SourceOnboardingService
            if not isinstance(self.db, dict) or KB_SOURCE_ADAPTERS in self.db:
                registered = await SourceOnboardingService(self.db).active_adapters()
                known = {adapter.name for adapter in adapters}
                adapters.extend(adapter for adapter in registered if adapter.name not in known)
        for adapter in adapters:
            if not _adapter_in_scope(adapter, jurisdiction_codes):
                continue
            state = await self.state.find_one({"_id": adapter.name}) or {}
            try:
                candidates, checkpoint = await adapter.discover(state.get("checkpoint"))
                # A previously productive, single-URL catalogue that suddenly
                # matches nothing from an otherwise-successful fetch is almost
                # always a redesigned portal breaking our selectors, not an
                # empty listing -- silently recording it as a clean empty run
                # would hide exactly that. Multi-URL paginated adapters (e.g.
                # `IndiaCodeAdapter`'s ~34 offset pages) legitimately rotate
                # through pages with naturally varying, sometimes zero, counts
                # and are excluded from this check.
                urls = getattr(getattr(adapter, "config", None), "urls", ())
                if not candidates and len(urls) <= 1 and int(state.get("last_discovered_count") or 0) > 0:
                    raise PortalLayoutChangedError(
                        f"{adapter.name} previously discovered "
                        f"{state.get('last_discovered_count')} candidate(s) and now discovered 0 "
                        "from an otherwise-successful fetch."
                    )
                candidates = [
                    candidate for candidate in candidates
                    if _candidate_in_scope(candidate, jurisdiction_codes)
                ]
                for candidate in candidates:
                    now = datetime.now(UTC)
                    result = await self.jobs.update_one(
                        {"source_key": candidate.source_key},
                        {"$setOnInsert": {
                            "_id": str(uuid4()), "source_key": candidate.source_key,
                            "candidate": asdict(candidate),
                            "canonical_document_key": candidate.canonical_document_key,
                            "status": "pending", "attempts": 0, "created_at": now,
                            "next_attempt_at": now,
                        }}, upsert=True,
                    )
                    counts["discovered" if result.upserted_id else "already_known"] += 1
                await self.state.update_one(
                    {"_id": adapter.name}, {"$set": {
                        "checkpoint": checkpoint, "last_success_at": datetime.now(UTC),
                        "last_discovered_count": len(candidates), "last_error": None,
                        "layout_changed": False,
                    }},
                    upsert=True,
                )
            except PortalLayoutChangedError as exc:
                counts["adapter_failures"] += 1
                await self.state.update_one(
                    {"_id": adapter.name}, {"$set": {
                        "last_failure_at": datetime.now(UTC), "last_error": type(exc).__name__,
                        "layout_changed": True,
                    }}, upsert=True,
                )
                await self._alert("kb_adapter_layout_changed", adapter=adapter.name, error=str(exc))
            except Exception as exc:  # noqa: BLE001 - one adapter cannot stop others
                counts["adapter_failures"] += 1
                await self.state.update_one(
                    {"_id": adapter.name}, {"$set": {
                        "last_failure_at": datetime.now(UTC), "last_error": type(exc).__name__,
                    }}, upsert=True,
                )
                await self._alert("kb_adapter_failed", adapter=adapter.name, error=type(exc).__name__)
        return counts

    async def run_due(
        self, limit: int = 20, jurisdiction_codes: tuple[str, ...] = (),
    ) -> dict[str, int]:
        counts = {
            "processed": 0, "published": 0, "quarantined": 0, "duplicate": 0, "retry": 0,
            "manual_access_required": 0,
        }
        for _ in range(limit):
            job = await self._claim(jurisdiction_codes)
            if job is None:
                break
            outcome = await self._process_claimed(job)
            counts["processed"] += 1
            counts[outcome] += 1
        return counts

    async def run_cycle(
        self, limit: int = 20, jurisdiction_codes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        await self.ensure_indexes()
        return {
            "scope": list(jurisdiction_codes) or ["all"],
            "discovery": await self.discover(jurisdiction_codes),
            "change_detection": await self.probe_due_sources(limit, jurisdiction_codes),
            "processing": await self.run_due(limit, jurisdiction_codes),
            "publication_benchmark": await self.promote_approved(limit, jurisdiction_codes),
            "canary_revalidation": await self.revalidate_published(limit, jurisdiction_codes),
        }

    async def probe_due_sources(
        self, limit: int = 20, jurisdiction_codes: tuple[str, ...] = (),
    ) -> dict[str, int]:
        """Re-hash known official URLs so silent replacements create versions."""
        now = datetime.now(UTC)
        counts = {"checked": 0, "changed": 0, "unchanged": 0, "failed": 0}
        query: dict[str, Any] = {
            "status": {"$in": ["published", "quarantined"]},
            "checksum": {"$exists": True},
            "$or": [{"next_source_check_at": {"$exists": False}}, {"next_source_check_at": {"$lte": now}}],
        }
        if jurisdiction_codes:
            query = {"$and": [query, _scope_query(jurisdiction_codes)]}
        cursor = self.jobs.find(query).sort("next_source_check_at", 1).limit(limit)
        for job in [item async for item in cursor]:
            counts["checked"] += 1
            next_check = now + timedelta(hours=max(1, settings.kb_automation_source_refresh_hours))
            try:
                body, _, final_url = await self.downloader.fetch(job["candidate"]["url"])
                checksum = hashlib.sha256(body).hexdigest()
                if checksum == job["checksum"]:
                    counts["unchanged"] += 1
                else:
                    old = SourceCandidate(**job["candidate"])
                    changed = SourceCandidate(**{
                        **asdict(old), "version": f"{old.version}-content-{checksum[:12]}", "url": final_url,
                    })
                    result = await self.jobs.update_one(
                        {"source_key": changed.source_key},
                        {"$setOnInsert": {
                            "_id": str(uuid4()), "source_key": changed.source_key,
                            "candidate": asdict(changed),
                            "canonical_document_key": changed.canonical_document_key,
                            "status": "pending", "attempts": 0, "created_at": now,
                            "next_attempt_at": now, "detected_from_job_id": job["_id"],
                        }}, upsert=True,
                    )
                    counts["changed"] += int(bool(result.upserted_id))
                await self.jobs.update_one(
                    {"_id": job["_id"]}, {"$set": {"last_source_check_at": now,
                                                      "next_source_check_at": next_check}},
                )
            except Exception as exc:  # noqa: BLE001 - serving version remains unchanged
                counts["failed"] += 1
                await self.jobs.update_one(
                    {"_id": job["_id"]}, {"$set": {"last_source_check_at": now,
                                                      "next_source_check_at": next_check,
                                                      "source_check_error": type(exc).__name__}},
                )
                await self._alert("kb_source_change_check_failed", job_id=job["_id"],
                                  error=type(exc).__name__)
        return counts

    async def promote_approved(
        self, limit: int = 20, jurisdiction_codes: tuple[str, ...] = (),
    ) -> dict[str, int]:
        """Publish only approved documents that pass the real retrieval path."""
        counts = {"checked": 0, "published": 0, "failed": 0}
        query: dict[str, Any] = {
            "status": "quarantined", "document_id": {"$exists": True},
            "last_error": {"$exists": False},
        }
        if jurisdiction_codes:
            query = {"$and": [query, _scope_query(jurisdiction_codes)]}
        cursor = self.jobs.find(query).sort("updated_at", 1).limit(limit)
        async for job in cursor:
            if (job.get("candidate") or {}).get("change_type") in {
                "amendment", "repeal", "commencement",
            } and not job.get("relationship_reviewed_at"):
                continue
            document_id = job["document_id"]
            total = await self.db[EMBEDDINGS_METADATA].count_documents({"document_id": document_id})
            approved = await self.db[EMBEDDINGS_METADATA].count_documents({
                "document_id": document_id, "metadata.review_status": "approved",
            })
            if total <= 0 or approved != total:
                continue
            counts["checked"] += 1
            benchmark = await self.benchmark.evaluate(
                SourceCandidate(**job["candidate"]), job["source_document"],
            )
            status = "published" if benchmark.get("passed") else "quarantined"
            result = await self.jobs.update_one(
                {"_id": job["_id"], "status": "quarantined"},
                {"$set": {"status": status, "retrieval_benchmark": benchmark,
                           "benchmarked_at": datetime.now(UTC),
                           "next_canary_at": datetime.now(UTC) + timedelta(
                               hours=max(1, settings.kb_canary_recheck_hours)
                           )}},
            )
            if result.modified_count:
                counts["published" if status == "published" else "failed"] += 1
        return counts

    async def record_relationship_review(
        self, job_id: str, related_job_ids: list[str], notes: str, actor: str,
    ) -> dict[str, Any]:
        job = await self.jobs.find_one({"_id": job_id})
        if job is None:
            raise ValueError("Automation job not found.")
        change_type = (job.get("candidate") or {}).get("change_type")
        if change_type not in {"amendment", "repeal", "commencement"}:
            raise ValueError("This candidate does not require a legal relationship review.")
        unique_ids = list(dict.fromkeys(related_job_ids))
        found = await self.jobs.count_documents({"_id": {"$in": unique_ids}})
        if found != len(unique_ids):
            raise ValueError("Every related job must exist before relationship review.")
        now = datetime.now(UTC)
        await self.jobs.update_one({"_id": job_id}, {"$set": {
            "relationship_reviewed_at": now, "relationship_reviewed_by": actor,
            "related_job_ids": unique_ids, "relationship_review_notes": notes,
        }})
        await self._alert(
            "kb_legal_relationship_reviewed", job_id=job_id,
            related_job_ids=unique_ids, actor_user_id=actor,
        )
        return {
            "job_id": job_id, "change_type": change_type,
            "related_job_ids": unique_ids, "status": "reviewed",
        }

    async def revalidate_published(
        self, limit: int = 20, jurisdiction_codes: tuple[str, ...] = (),
    ) -> dict[str, int]:
        """Re-run retrieval canaries and fail closed if a published document regresses."""
        now = datetime.now(UTC)
        query: dict[str, Any] = {
            "status": "published",
            "$or": [{"next_canary_at": {"$exists": False}}, {"next_canary_at": {"$lte": now}}],
        }
        if jurisdiction_codes:
            query = {"$and": [query, _scope_query(jurisdiction_codes)]}
        counts = {"checked": 0, "passed": 0, "rolled_back": 0}
        async for job in self.jobs.find(query).sort("next_canary_at", 1).limit(limit):
            counts["checked"] += 1
            benchmark = await self.benchmark.evaluate(
                SourceCandidate(**job["candidate"]), job["source_document"],
            )
            next_canary = now + timedelta(hours=max(1, settings.kb_canary_recheck_hours))
            if benchmark.get("passed"):
                counts["passed"] += 1
                await self.jobs.update_one({"_id": job["_id"], "status": "published"}, {"$set": {
                    "retrieval_benchmark": benchmark, "benchmarked_at": now,
                    "next_canary_at": next_canary,
                }})
                continue
            document_id = job.get("document_id")
            if document_id:
                await self.db[EMBEDDINGS_METADATA].update_many(
                    {"document_id": document_id}, {"$set": {
                        "metadata.review_status": "needs_review",
                        "metadata.document_status": "quarantined",
                    }},
                )
            result = await self.jobs.update_one(
                {"_id": job["_id"], "status": "published"}, {"$set": {
                    "status": "quarantined", "retrieval_benchmark": benchmark,
                    "benchmarked_at": now, "last_error": "CanaryBenchmarkRegression",
                }},
            )
            if result.modified_count:
                from app.cache.response_cache import response_cache
                await response_cache.bump_generation(strict=True)
                counts["rolled_back"] += 1
                await self._alert("kb_canary_rollback", job_id=job["_id"],
                                  document_id=document_id)
        return counts

    async def retry(self, job_id: str) -> bool:
        result = await self.jobs.update_one(
            {"_id": job_id, "$or": [
                {"status": "retry"},
                {"status": "quarantined", "last_error": {"$exists": True}},
            ]},
            {"$set": {"status": "pending", "next_attempt_at": datetime.now(UTC), "last_error": None}},
        )
        if result.modified_count:
            return True
        # Security/correctness finding G5: a 0-match update means either "this
        # job exists but isn't in a retryable state" (a real, non-error
        # `requeued: false`) or "this job_id was never registered at all" --
        # conflating them reported 200 `requeued: false` for a nonexistent
        # job, indistinguishable from a legitimate "nothing to retry" result.
        if await self.jobs.find_one({"_id": job_id}) is None:
            raise NotFoundError(f"Automation job '{job_id}' not found.")
        return False

    async def coverage(self) -> dict[str, Any]:
        statuses = [
            "pending", "processing", "retry", "duplicate", "quarantined", "published",
            "manual_access_required",
        ]
        counts = {status: await self.jobs.count_documents({"status": status}) for status in statuses}
        failed_adapters = await self.events.count_documents({"event_type": "kb_adapter_failed"})
        layout_changed_adapters = await self.events.count_documents({"event_type": "kb_adapter_layout_changed"})
        oldest_manual_access = None
        async for job in self.jobs.find({"status": "manual_access_required"}).sort("updated_at", 1).limit(1):
            oldest_manual_access = job
        adapter_rows = []
        for adapter in self.adapters:
            state = await self.state.find_one({"_id": adapter.name}) or {}
            adapter_rows.append({
                "adapter": adapter.name,
                "status": "layout_changed" if state.get("layout_changed") else (
                    "failed" if state.get("last_error") else (
                        "empty" if state.get("last_success_at") and not state.get("last_discovered_count") else (
                            "healthy" if state.get("last_success_at") else "never_run"
                        )
                    )
                ),
                "last_success_at": state.get("last_success_at"),
                "last_failure_at": state.get("last_failure_at"),
                "last_discovered_count": state.get("last_discovered_count", 0),
                "checkpoint": state.get("checkpoint"),
            })
        return {
            "enabled": settings.kb_automation_enabled,
            "stages": counts,
            "adapter_failures": failed_adapters,
            # This is the honest count backing "manual downloading stays
            # limited to genuinely inaccessible-portal exceptions" -- anything
            # NOT in this bucket is expected to resolve through retry/backoff
            # without a human touching it.
            "routine_manual_downloads_required": counts["manual_access_required"],
            "exception_queue": counts["quarantined"],
            "dead_letter": {
                "quarantined": counts["quarantined"],
                "manual_access_required": counts["manual_access_required"],
                "oldest_manual_access_required": oldest_manual_access,
                "layout_changed_adapters": layout_changed_adapters,
            },
            "adapters": adapter_rows,
        }

    async def per_jurisdiction_summary(self) -> dict[str, dict[str, int]]:
        """Job counts grouped by State/UT code, for
        `KnowledgeBaseCoverageService.merge_automation`. A job counts toward
        every code it actually binds (`jurisdiction_code` plus any extra
        `applicable_state_codes`, e.g. a multi-state High Court source) --
        "IN" (Central) is excluded, since the chunk-derived matrix already
        covers Central separately."""
        summary: dict[str, dict[str, int]] = {}
        downloaded_statuses = {
            "pending", "processing", "retry", "duplicate", "quarantined",
            "published", "manual_access_required",
        }
        projection = {
            "candidate.jurisdiction_code": 1, "candidate.applicable_state_codes": 1,
            "status": 1, "retrieval_benchmark.passed": 1,
        }
        async for job in self.jobs.find({}, projection):
            candidate = job.get("candidate") or {}
            codes = {candidate.get("jurisdiction_code"), *(candidate.get("applicable_state_codes") or [])}
            codes.discard(None)
            codes.discard("IN")
            if not codes:
                continue
            status = job.get("status")
            benchmark = job.get("retrieval_benchmark")
            for code in codes:
                row = summary.setdefault(code, {
                    "discovered": 0, "downloaded": 0, "manual_access_required": 0,
                    "tested_passed": 0, "tested_total": 0,
                })
                row["discovered"] += 1
                if status in downloaded_statuses:
                    row["downloaded"] += 1
                if status == "manual_access_required":
                    row["manual_access_required"] += 1
                # A benchmark was actually executed (`promote_approved`), pass
                # or fail -- distinct from `status == "published"`, which only
                # a PASSING benchmark produces (a failing one reverts to
                # "quarantined"). Both outcomes belong in `tested_total`.
                if benchmark is not None:
                    row["tested_total"] += 1
                    row["tested_passed"] += int(bool(benchmark.get("passed")))
        return summary

    async def list_jobs(
        self, status: str | None = None, limit: int = 100, order_by_confidence: bool = False,
    ) -> list[dict[str, Any]]:
        query = {"status": status} if status else {}
        # Confidence ordering surfaces the most promising review candidates
        # first; it is purely a queue-ordering convenience and never touched
        # by ingestion/verification decisions themselves.
        sort_key = "candidate.confidence_score" if order_by_confidence else "created_at"
        return [item async for item in self.jobs.find(query).sort(sort_key, -1).limit(limit)]

    async def _claim(self, jurisdiction_codes: tuple[str, ...] = ()) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        token = str(uuid4())
        query: dict[str, Any] = {
            "status": {"$in": ["pending", "retry", "processing"]}, "next_attempt_at": {"$lte": now},
            "$or": [{"lease_until": {"$exists": False}}, {"lease_until": {"$lte": now}}],
        }
        if jurisdiction_codes:
            query = {"$and": [query, _scope_query(jurisdiction_codes)]}
        claimed = await self.jobs.find_one_and_update(
            query,
            {"$set": {"status": "processing", "lease_token": token,
                       "lease_until": now + timedelta(minutes=10), "started_at": now}},
            sort=[("next_attempt_at", 1)], return_document=ReturnDocument.AFTER,
        )
        return claimed if isinstance(claimed, dict) else None

    async def _process_claimed(self, job: dict[str, Any]) -> str:
        candidate = SourceCandidate(**job["candidate"])
        temp_dir = settings.kb_staging_dir / "automation_downloads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{job['_id']}.download"
        try:
            body, content_type, final_url = await self.downloader.fetch(candidate.url)
            checksum = hashlib.sha256(body).hexdigest()
            duplicate = await self.jobs.find_one({
                "_id": {"$ne": job["_id"]}, "checksum": checksum,
                "status": {"$in": ["published", "quarantined", "duplicate"]},
            })
            if duplicate:
                await self._finish(job, "duplicate", checksum=checksum, duplicate_of=duplicate["_id"])
                return "duplicate"
            previous_version = await self.jobs.find_one(
                {"_id": {"$ne": job["_id"]},
                 "canonical_document_key": candidate.canonical_document_key,
                 "candidate.language": candidate.language,
                 "status": {"$in": ["published", "quarantined"]}},
                sort=[("created_at", -1)],
            )
            language_sibling = await self.jobs.find_one(
                {"_id": {"$ne": job["_id"]},
                 "canonical_document_key": candidate.canonical_document_key,
                 "candidate.language": {"$ne": candidate.language},
                 "status": {"$in": ["published", "quarantined"]}},
                sort=[("created_at", -1)],
            )
            temp_path.write_bytes(body)
            await self.scanner.scan(temp_path)
            is_state_specific = candidate.jurisdiction_code != "IN"
            metadata = document_metadata_fields(normalize_jurisdiction({
                "document_key": candidate.canonical_document_key,
                "issuing_level": "central" if candidate.jurisdiction_code == "IN" else "state",
                # A structural fact the adapter already has evidence for, not
                # a verification claim -- `verification_status` below stays
                # "unverified" regardless. Per `_review_reasons`, an "unknown"
                # applicability is itself a reason a record needs review, so
                # supplying the real one here is a strict improvement.
                "applicability": "specific_states" if is_state_specific else "unknown",
                "applicable_state_codes": (
                    sorted({candidate.jurisdiction_code, *candidate.applicable_state_codes})
                    if is_state_specific else []
                ),
                "jurisdiction_source_type": candidate.document_type,
                "source_url": final_url,
                "version_label": candidate.version,
                "effective_from": candidate.effective_from,
                "verification_status": "unverified",
                "metadata_provenance": "automated_official",
                "machine_verification": {
                    "download_sha256": checksum,
                    "downloaded_at": datetime.now(UTC).isoformat(),
                    "source_language": candidate.language,
                    "issuing_authority": candidate.issuing_authority,
                    "publication_date": candidate.publication_date,
                    "parent_external_id": candidate.parent_external_id,
                },
            }, provenance="automated_official"))
            extension = ".pdf" if content_type == _PDF else ".html"
            safe_filename = f"{Path(candidate.filename).stem}{extension}"
            upload = UploadFile(
                file=io.BytesIO(body), filename=safe_filename,
                headers=Headers({"content-type": content_type}),
            )
            staged = await self.ingestion.stage(upload, jurisdiction_metadata=metadata)
            if not staged.claimed:
                await self._finish(job, "duplicate", checksum=checksum, staging_id=staged.staging_id)
                return "duplicate"
            result = await self.ingestion.process(staged.staging_id, staged.staged_path, staged.original_filename)
            if result.status != "indexed" or not result.document_id or not result.chunks_indexed:
                raise ValueError(result.reason or "Canary ingestion did not produce a complete searchable version.")
            status = "published" if result.review_status == "approved" else "quarantined"
            await self._finish(
                job, status, checksum=checksum, final_url=final_url, staging_id=staged.staging_id,
                document_id=result.document_id, chunks_indexed=result.chunks_indexed,
                source_document=result.generated_filename or candidate.filename,
                supersedes_job_id=previous_version.get("_id") if previous_version else None,
                translation_of_job_id=language_sibling.get("_id") if language_sibling else None,
                reason=None if status == "published" else "Indexed but hidden pending legal verification.",
            )
            return status
        except ManualAccessRequiredError as exc:
            # A CAPTCHA/login wall cannot be beaten by retrying on a timer --
            # this is a terminal state distinct from "quarantined" so an
            # operator can find exactly the portals that need a human to
            # fetch the document by hand and upload it through the existing
            # knowledge-gap source flow. `retry()` explicitly refuses it.
            await self._finish(
                job, "manual_access_required", attempts=int(job.get("attempts", 0)) + 1,
                last_error=type(exc).__name__, error_detail=str(exc)[:1000],
            )
            await self._alert("kb_manual_access_required", job_id=job["_id"], url=candidate.url)
            return "manual_access_required"
        except Exception as exc:  # noqa: BLE001 - durable retry/quarantine is the contract
            attempts = int(job.get("attempts", 0)) + 1
            terminal = attempts >= settings.kb_automation_max_attempts
            status = "quarantined" if terminal else "retry"
            await self._finish(
                job, status, attempts=attempts, last_error=type(exc).__name__, error_detail=str(exc)[:1000],
                next_attempt_at=datetime.now(UTC) + timedelta(minutes=min(1440, 2 ** attempts * 5)),
            )
            await self._alert("kb_automation_job_failed", job_id=job["_id"], status=status,
                              error=type(exc).__name__)
            return status
        finally:
            temp_path.unlink(missing_ok=True)

    async def _finish(self, job: dict[str, Any], status: str, **fields: Any) -> None:
        await self.jobs.update_one(
            {"_id": job["_id"], "lease_token": job["lease_token"]},
            {"$set": {"status": status, "updated_at": datetime.now(UTC), **fields},
             "$unset": {"lease_token": "", "lease_until": ""}},
        )

    async def _alert(self, event_type: str, **details: Any) -> None:
        await self.events.insert_one({"event_type": event_type, "details": details, "created_at": datetime.now(UTC)})


class KnowledgeBaseAutomationScheduler:
    def __init__(self, service: KnowledgeBaseAutomationService | None = None) -> None:
        # MongoDB is connected inside FastAPI lifespan, after module imports.
        # Delay the default service until start() so importing app.main never
        # touches an unconnected database handle.
        self.service = service
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if settings.kb_automation_enabled and (self.task is None or self.task.done()):
            self.service = self.service or KnowledgeBaseAutomationService()
            self.task = asyncio.create_task(self._loop())

    async def close(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def _loop(self) -> None:
        if self.service is None:
            raise RuntimeError("KB automation scheduler started without a service.")
        while True:
            try:
                await self.service.run_cycle()
                from app.services.kb_operations import KBOperationsService
                await KBOperationsService(self.service.db).audit_and_alert()
            except Exception as exc:  # noqa: BLE001
                log.error("kb_automation_cycle_failed", error=type(exc).__name__)
            await asyncio.sleep(max(60, settings.kb_automation_interval_minutes * 60))


kb_automation_scheduler = KnowledgeBaseAutomationScheduler()
