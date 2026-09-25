"""Auto-fetch a likely-missing Act when a chat answer comes back out-of-KB.

Opt-in via `settings.kb_gap_autofetch_enabled` -- this reaches out to the
internet based on live, unauthenticated user input, so an operator must
deliberately turn it on.

Flow: `record_gap()` is fired-and-forgotten from the chat path (never
awaited, never allowed to affect the response already sent) and only
extracts a candidate Act name and enqueues it. `GapAutoFetchScheduler` drains
the queue on its own schedule; nothing here ever runs synchronously inside a
chat request.

Every step fails closed:
  - no LLM key / no search results / no .gov.in-.nic.in candidate -> the job
    just stays pending for a later retry, up to `kb_gap_autofetch_max_attempts`.
  - a candidate that fetches but isn't a real PDF, or whose extracted text
    doesn't contain the Act's own name -> rejected, next candidate tried.
  - identity matches but the applicability/commencement phrase banks don't
    -> still indexed (so a human reviewing the gap doesn't start from
    nothing), but left `needs_review` exactly like every hand-curated source
    in this project that couldn't prove the fuller bar.
  - identity + applicability + commencement all match -> published via the
    same `verification_status="machine_verified"` evidence-bundle path as
    `app/services/kb_machine_verification.py`'s hand-curated policies, which
    is the only way automated code is allowed to reach `review_status=
    "approved"` in this codebase (see `app/rag/kb_jurisdiction.py`'s
    `_resolve_verification`/`_review_reasons` -- plain "verified" is a
    human-only status; that rule is not relaxed here, it is *satisfied*).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import UploadFile

from app.core.config import settings
from app.database.mongodb import mongodb
from app.llm.base import ChatMessage
from app.llm.factory import LLMFactory
from app.models.collections import KB_GAP_AUTOFETCH_QUEUE
from app.rag.kb_jurisdiction import (
    PROVENANCE_AUTOMATED_OFFICIAL,
    JurisdictionMetadataError,
    document_metadata_fields,
    normalize_jurisdiction,
)
from app.schemas.law_monitoring import official_url
from app.services.kb_automation import ManualAccessRequiredError, OfficialDownloader
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService
from app.services.kb_machine_verification import normalized_text
from app.services.kb_presence import find_existing_kb_law
from app.utils.malware_scan import ClamAVScanner, MalwareScanner

log = structlog.get_logger(__name__)

_ACT_NAME_PROMPT = (
    "You will be shown a legal question asked in India. Reply with ONLY the official "
    'short title (including the year, e.g. "Consumer Protection Act, 2019") of the '
    "single Indian central Act, Code, or Rules this question is most likely about. "
    "If you cannot confidently identify one specific Act, reply with exactly: NONE. "
    "Do not explain your answer. Do not answer the legal question itself."
)

# Boilerplate that appears, close to verbatim, in the "Extent" clause of most
# Indian central Acts when they apply nationwide. Presence of ANY one is
# treated as applicability evidence; ABSENCE of all of them is not treated as
# proof the Act is NOT all-India -- it just fails this check, same
# fail-closed rule as everywhere else in this module. A State-specific or
# model law (see `app/rag/kb_jurisdiction.py`'s own warning about the Model
# Tenancy Act, 2021) is expected to fail this check, never to be talked into
# passing it.
_APPLICABILITY_PHRASES = (
    "extendstothewholeofindia",
    "shallextendtothewholeofindia",
    "applytothewholeofindia",
    "shallapplytothewholeofindia",
)

# The commencement clause every Indian central Act carries in some form.
_COMMENCEMENT_PHRASES = (
    "shallcomeintoforce",
    "comeintoforceonsuchdate",
    "shallcomeintooperation",
)


@dataclass(frozen=True)
class _Attempt:
    review_status: str
    metadata: dict[str, Any]


class GapAutoFetchService:
    def __init__(
        self, db: Any = None, downloader: Any = None, ingestion: Any = None,
        scanner: MalwareScanner | None = None, llm_factory: type[LLMFactory] = LLMFactory,
    ) -> None:
        # `db` is resolved lazily (see `queue` below), never here: this
        # constructor is called from `_no_verified_context_answer` --
        # a synchronous method exercised directly by unit tests with no live
        # Mongo connection -- and must stay constructible with no active
        # database, exactly like the fire-and-forget call site that creates
        # it promises never to affect the chat response either way.
        self._db_override = db
        self.downloader = downloader or OfficialDownloader()
        self.ingestion = ingestion or KnowledgeBaseIngestionService()
        self.scanner = scanner or ClamAVScanner()
        self.llm_factory = llm_factory

    @property
    def db(self) -> Any:
        return self._db_override if self._db_override is not None else mongodb.db

    @property
    def queue(self) -> Any:
        return self.db[KB_GAP_AUTOFETCH_QUEUE]

    async def ensure_indexes(self) -> None:
        await self.queue.create_index("act_name", unique=True)
        await self.queue.create_index([("status", 1), ("next_attempt_at", 1)])

    # ------------------------------------------------------------ detection

    async def record_gap(self, query: str, message_id: str | None = None) -> None:
        """Fire-and-forget from the chat path. Never raises -- a failure here
        must never surface as, or affect, the chat response it follows."""
        if not settings.kb_gap_autofetch_enabled:
            return
        try:
            act_name = await self._extract_act_name(query)
            if not act_name:
                return
            # An Act the KB already holds is a retrieval/bridge miss, not a
            # gap: queueing it would try to download something already indexed.
            presence = await find_existing_kb_law(self.db, act_name)
            if presence is not None:
                log.info(
                    "kb_gap_autofetch_skipped_existing", act_name=act_name, law=presence.law_key,
                    status=presence.status,
                )
                return
            now = datetime.now(UTC)
            await self.queue.update_one(
                {"act_name": act_name},
                {
                    "$setOnInsert": {
                        "act_name": act_name, "status": "pending", "attempts": 0,
                        "created_at": now, "next_attempt_at": now,
                        "sample_query": query[:500], "sample_message_id": message_id,
                    },
                },
                upsert=True,
            )
        except Exception as exc:  # noqa: BLE001 - background best-effort, must never surface to the chat response
            log.warning("kb_gap_autofetch_record_failed", error=type(exc).__name__)

    async def _extract_act_name(self, query: str) -> str | None:
        llm = self.llm_factory.create_resilient()
        response = await llm.chat(
            [
                ChatMessage(role="system", content=_ACT_NAME_PROMPT),
                ChatMessage(role="user", content=query),
            ],
            temperature=0.0,
        )
        if response.error or not response.content:
            return None
        text = response.content.strip().strip('"').strip()
        if not text or text.upper() == "NONE" or len(text) > 200:
            return None
        return text

    # ------------------------------------------------------------- draining

    async def process_due(self, limit: int = 3) -> dict[str, int]:
        counts = {"processed": 0, "published": 0, "needs_review": 0, "failed": 0, "skipped_existing": 0}
        if not settings.kb_gap_autofetch_enabled:
            return counts
        for _ in range(limit):
            job = await self._claim()
            if job is None:
                break
            outcome = await self._process(job)
            counts["processed"] += 1
            counts[outcome] += 1
        return counts

    async def _claim(self) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        claimed: dict[str, Any] | None = await self.queue.find_one_and_update(
            {"status": "pending", "next_attempt_at": {"$lte": now}},
            {"$set": {"status": "processing", "claimed_at": now}, "$inc": {"attempts": 1}},
            sort=[("created_at", 1)],
        )
        return claimed

    async def _process(self, job: dict[str, Any]) -> str:
        act_name = job["act_name"]
        # Jobs queued before `record_gap` grew the presence check may name an
        # Act that is already indexed; close them out instead of fetching.
        presence = await find_existing_kb_law(self.db, act_name)
        if presence is not None:
            await self._finish(
                job, "skipped_existing",
                reason=f"Already covered by the Knowledge Base ({presence.law_key}: {presence.status}); "
                "this was a retrieval miss, not a corpus gap.",
            )
            return "skipped_existing"
        try:
            candidates = await self._search_candidates(act_name)
        except Exception as exc:  # noqa: BLE001 - a search failure just fails this attempt, retried later
            candidates = []
            log.warning("kb_gap_autofetch_search_failed", act_name=act_name, error=type(exc).__name__)
        for url in candidates:
            try:
                attempt = await self._try_candidate(act_name, url)
            except Exception as exc:  # noqa: BLE001 - one bad candidate must not sink the whole job
                log.warning(
                    "kb_gap_autofetch_candidate_failed", act_name=act_name, url=url, error=type(exc).__name__,
                )
                continue
            if attempt is None:
                continue
            await self._finish(job, attempt.review_status, source_url=url)
            return attempt.review_status
        if job["attempts"] >= settings.kb_gap_autofetch_max_attempts:
            await self._finish(job, "failed", reason="No verifiable official source found after max attempts.")
            return "failed"
        await self.queue.update_one(
            {"_id": job["_id"]}, {"$set": {"status": "pending", "next_attempt_at": datetime.now(UTC)}},
        )
        return "failed"

    async def _search_candidates(self, act_name: str) -> list[str]:
        gemini = self.llm_factory.create("gemini")
        search = getattr(gemini, "search_grounded_urls", None)
        if search is None:
            return []
        query = f"{act_name} official text pdf site:gov.in OR site:nic.in"
        urls = await search(query)
        verified: list[str] = []
        for url in urls:
            try:
                official_url(url)  # https, .gov.in/.nic.in only, no port/credentials/fragment
            except ValueError:
                continue
            verified.append(url)
        return verified

    async def _try_candidate(self, act_name: str, url: str) -> _Attempt | None:
        try:
            body, content_type, _final_url = await self.downloader.fetch(url)
        except ManualAccessRequiredError:
            return None
        if content_type != "application/pdf" or not body.startswith(b"%PDF"):
            return None
        text = normalized_text(body)
        if not text or _normalize_title(act_name) not in text:
            return None

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(body)
            scan_path = Path(handle.name)
        try:
            await self.scanner.scan(scan_path)
        finally:
            scan_path.unlink(missing_ok=True)

        applicability_hit = any(p in text for p in _APPLICABILITY_PHRASES)
        commencement_hit = any(p in text for p in _COMMENCEMENT_PHRASES)
        official_hash = hashlib.sha256(body).hexdigest()
        document_key = _slugify(act_name)
        filename = f"AUTOFETCH_{document_key.upper().replace('-', '_')}.pdf"

        if applicability_hit and commencement_hit:
            evidence = {
                "official_url": url, "official_sha256": official_hash, "local_sha256": official_hash,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "policy_version": "gap-autofetch-v1", "verified_at": datetime.now(UTC).isoformat(),
                "identity_checks": [True], "applicability_checks": [True], "commencement_checks": [True],
                "commencement_url": url, "commencement_sha256": official_hash, "exact_byte_match": True,
            }
            raw = {
                "document_key": document_key, "issuing_level": "central", "applicability": "all_india",
                "jurisdiction_source_type": "bare_act", "source_url": url,
                "version_label": f"Auto-fetched for the question about '{act_name}'; identity, "
                "applicability, and commencement phrases all verified against the fetched text.",
                "verification_status": "machine_verified", "machine_verification": evidence,
            }
        else:
            raw = {
                "document_key": document_key, "issuing_level": "central", "applicability": "unknown",
                "jurisdiction_source_type": "bare_act", "source_url": url,
                "version_label": f"Auto-fetched for the question about '{act_name}'; the Act's own name "
                "matched but the applicability/commencement phrase check did not -- needs human review "
                "before this is treated as confirmed, current, all-India law.",
                "verification_status": "unverified",
            }
        try:
            normalized = normalize_jurisdiction(raw, provenance=PROVENANCE_AUTOMATED_OFFICIAL)
        except JurisdictionMetadataError:
            return None
        metadata = document_metadata_fields(normalized)
        upload = UploadFile(file=io.BytesIO(body), filename=filename)
        response = await self.ingestion.ingest(upload, jurisdiction_metadata=metadata)
        if response.status not in ("indexed", "duplicate"):
            return None
        return _Attempt(review_status=metadata["review_status"], metadata=metadata)

    async def _finish(
        self, job: dict[str, Any], status: str, *, source_url: str | None = None, reason: str | None = None,
    ) -> None:
        await self.queue.update_one(
            {"_id": job["_id"]},
            {"$set": {"status": status, "finished_at": datetime.now(UTC), "source_url": source_url, "reason": reason}},
        )


def _normalize_title(act_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", act_name.lower())


def _slugify(act_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", act_name.lower()).strip("-")
    return slug or "auto-fetched-act"


class GapAutoFetchScheduler:
    def __init__(self, service: GapAutoFetchService | None = None) -> None:
        # Mirrors `KnowledgeBaseAutomationScheduler` (app/services/
        # kb_automation.py): MongoDB connects inside FastAPI lifespan, after
        # module imports, so the default service is created lazily in
        # `start()` rather than at import time.
        self.service = service
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if settings.kb_gap_autofetch_enabled and (self.task is None or self.task.done()):
            self.service = self.service or GapAutoFetchService()
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
            raise RuntimeError("Gap auto-fetch scheduler started without a service.")
        await self.service.ensure_indexes()
        while True:
            try:
                await self.service.process_due()
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the scheduler loop
                log.error("kb_gap_autofetch_cycle_failed", error=type(exc).__name__)
            await asyncio.sleep(max(60, settings.kb_gap_autofetch_interval_minutes * 60))


gap_autofetch_scheduler = GapAutoFetchScheduler()
