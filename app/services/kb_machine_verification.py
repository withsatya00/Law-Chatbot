"""Fail-closed verification against exact official bytes and Gazette notices."""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from pypdf import PdfReader

from app.core.config import settings
from app.database.mongodb import mongodb
from app.models.collections import UPLOADED_DOCUMENTS
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService
from app.services.law_monitoring import fetch_official_snapshot

log = structlog.get_logger(__name__)

POLICY_VERSION = "official-exact-v1"
AUDIT_COLLECTION = "kb_machine_verification_runs"


def normalized_text(data: bytes) -> str:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


@dataclass(frozen=True)
class MachinePolicy:
    key: str
    filename: str
    official_url: str
    commencement_url: str
    document_key: str
    version_label: str
    identity_tokens: tuple[str, ...]
    applicability_tokens: tuple[str, ...]
    commencement_tokens: tuple[str, ...]
    # `None` for an Act old enough, and amended often enough, that a single
    # "effective_from" would misstate the CURRENT consolidated text as though
    # it were the original enactment (see the Contract Act / RTI / CPC
    # policies below) -- a legal-policy call the human reviewer already made
    # by leaving it unset, not something this module should compute or guess.
    effective_from: str | None = None
    section_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    excluded_text_patterns: tuple[str, ...] = ()


POLICIES = (
    MachinePolicy(
        key="BNS", filename="The_Bhara_Tiy_A_Ny_A_Y_A_Sanhita_2023_5.pdf",
        official_url="https://www.mha.gov.in/sites/default/files/2024-04/250883_english_01042024.pdf",
        commencement_url="https://www.mha.gov.in/sites/default/files/BhartiyaNyayaSanhita_24022024.pdf",
        document_key="bharatiya-nyaya-sanhita-2023", version_label="Act 45 of 2023 (as enacted)",
        identity_tokens=("bharatiyanyayasanhita2023", "no45of2023", "25thdecember2023"),
        applicability_tokens=("guiltywithinindia", "actcommittedbeyondindia", "citizenofindiainanyplacewithoutandbeyondindia"),
        commencement_tokens=("so850e", "1stdayofjuly2024", "subsection2ofsection106"),
        # The legacy chunks lack reliable section metadata. The distinctive
        # statutory phrase locates the chunk containing subsection 106(2);
        # excluding that whole chunk is deliberately conservative.
        effective_from="2024-07-01", excluded_text_patterns=("registered medical practitioner",),
    ),
    MachinePolicy(
        key="BSA", filename="The_Bhara_Tiy_A_Sakshy_A_Adhiniy_Am_2023_6.pdf",
        official_url="https://www.mha.gov.in/sites/default/files/2024-04/250882_english_01042024_0.pdf",
        commencement_url="https://www.mha.gov.in/sites/default/files/BharatiyaSakshyaAdhiniyam_24022024.pdf",
        document_key="bharatiya-sakshya-adhiniyam-2023", version_label="Act 47 of 2023 (as enacted)",
        identity_tokens=("bharatiyasakshyaadhiniyam2023", "no47of2023", "25thdecember2023"),
        applicability_tokens=("appliestoalljudicialproceedings", "includingcourtsmartial"),
        commencement_tokens=("1stdayofjuly2024", "bharatiyasakshyaadhiniyam2023"),
        effective_from="2024-07-01",
    ),
    MachinePolicy(
        key="CONTRACT_ACT", filename="Indian_Contract_Act_1872_CAG_Official.pdf",
        official_url="https://www.cag.gov.in/uploads/media/Indian-Contract-Act-1872-20200816140128.pdf",
        # No separate commencement notification exists for an 1872 Act; the
        # CAG's own consolidated text states its enactment commencement in
        # section 1, so that is the evidence used, same pattern as BSA.
        commencement_url="https://www.cag.gov.in/uploads/media/Indian-Contract-Act-1872-20200816140128.pdf",
        document_key="indian-contract-act-1872", version_label="Act 9 of 1872 (CAG compiled text, as amended)",
        identity_tokens=("indiancontractact1872", "no9of1872"),
        applicability_tokens=("extendstothewholeofindia",),
        commencement_tokens=("shallcomeintoforceonthefirstdayofseptember1872",),
        # No `effective_from`: this is a 150-year-old, heavily amended Act.
        # The 1872 commencement date governs the ORIGINAL enactment, not this
        # "as amended" consolidated text -- the human reviewer already left
        # this field unset (2026-09-08 review) rather than pick a date, and
        # this policy does not override that judgment.
    ),
    MachinePolicy(
        key="RTI_ACT", filename="RTI_Act_2005_Official_Amended.pdf",
        official_url="https://rti.dopt.gov.in/Writereaddata/RTI%20Act,%202005%20(Amended)-English%20Version.PDF",
        commencement_url="https://rti.dopt.gov.in/Writereaddata/RTI%20Act,%202005%20(Amended)-English%20Version.PDF",
        document_key="right-to-information-act-2005", version_label="Official amended English version",
        identity_tokens=("righttoinformationact2005", "no22of2005", "15thjune2005"),
        applicability_tokens=("extendstothewholeofindia",),
        # ss. 4, 5, 12, 13, 15, 16, 24, 27, 28 commenced immediately on
        # enactment; the rest (including the core RTI-request machinery)
        # 120 days later. No `effective_from`: the Act has since been amended
        # (the 2019 Act 24-of-2019 change is visible in the indexed text
        # itself), so a single date would misstate the current, as-amended
        # text -- same reasoning as the Contract Act policy above.
        commencement_tokens=("remainingprovisionsofthisactshallcomeintoforceontheonehundredandtwentiethdayofitsenactment",),
    ),
    MachinePolicy(
        key="CPC", filename="CPC_1908_Official.pdf",
        official_url="https://sclsc.gov.in/theme/front/pdf/ACTS%20FINAL/THE%20CODE%20OF%20CIVIL%20PROCEDURE,%201908.pdf",
        commencement_url="https://sclsc.gov.in/theme/front/pdf/ACTS%20FINAL/THE%20CODE%20OF%20CIVIL%20PROCEDURE,%201908.pdf",
        document_key="code-of-civil-procedure-1908", version_label="Act 5 of 1908 (SCLSC compiled text, as amended)",
        identity_tokens=("codeofcivilprocedure1908", "no5of1908"),
        applicability_tokens=("extendstothewholeofindia",),
        commencement_tokens=("shallcomeintoforceonthefirstdayofjanuary1909",),
        # No `effective_from`, for the same reason as the Contract Act above.
    ),
    # NOT machine-verifiable yet, deliberately left out rather than added
    # non-functional (checked 2026-09-21):
    #  - Negotiable Instruments Act, 1881: no working direct-PDF official
    #    mirror was found -- indiacode.nic.in's SPA migration (see this
    #    module's other comments and SOURCE_VERIFICATION_CHECKLIST.md) no
    #    longer serves a matching bitstream, and the indexed copy's own
    #    recorded source_url is an HTML "handle" landing page, not a PDF.
    #  - Payment of Wages Act, 1936 (Maharashtra-hosted, State-modified
    #    copy): bombayhighcourt.gov.in currently refuses the TLS handshake
    #    this project's downloader uses (legacy renegotiation disabled); and
    #    it is a State-modified text (`applicable_state_codes: [MH]`), not
    #    `all_india`, which this service's `verify()` does not yet support.
    #  - Code on Wages, 2019: the currently indexed file
    #    (`2589gi_P65_6.pdf`) is NOT byte-identical to the live
    #    labour.gov.in URL (confirmed 2026-09-21 -- a real edition/version
    #    drift, not a bug here), so a policy would only ever report "held".
    #    The Act's own text also has no single commencement date ("on such
    #    date as the Central Government may... appoint"), so a genuine
    #    commencement notification PDF would still be needed even once the
    #    byte-hash question is resolved.
    # Add a policy for any of these once a human has sourced a working
    # official mirror (and, for Payment of Wages, State-applicability
    # support is added to `verify()`).
)


class MachineVerificationService:
    def __init__(self, fetcher: Any = None, publisher: Any = None, db: Any = None) -> None:
        self.fetcher = fetcher or fetch_official_snapshot
        self.publisher = publisher or KnowledgeBaseIngestionService()
        self.db = db if db is not None else mongodb.db

    async def run(self, *, force: bool = False) -> dict[str, Any]:
        latest = await self.db[AUDIT_COLLECTION].find_one({}, sort=[("ran_at", -1)])
        latest_run = latest.get("ran_at") if latest else None
        if latest_run and latest_run.tzinfo is None:
            latest_run = latest_run.replace(tzinfo=UTC)
        if not force and latest and latest_run and latest_run > datetime.now(UTC) - timedelta(hours=24):
            return {
                "status": "not_due", "last_run_at": latest_run, "policy_version": POLICY_VERSION,
                "published": latest.get("published", 0), "held": latest.get("held", 0),
                "results": latest.get("results", []),
            }
        results = [await self.verify(policy) for policy in POLICIES]
        summary = {
            "policy_version": POLICY_VERSION, "ran_at": datetime.now(UTC),
            "published": sum(item["status"] == "published" for item in results),
            "held": sum(item["status"] != "published" for item in results), "results": results,
        }
        await self.db[AUDIT_COLLECTION].insert_one({"_id": str(datetime.now(UTC).timestamp()), **summary})
        return summary

    async def verify(self, policy: MachinePolicy) -> dict[str, Any]:
        result: dict[str, Any] = {"law": policy.key, "filename": policy.filename, "status": "held"}
        document = await self.db[UPLOADED_DOCUMENTS].find_one({"filename": policy.filename})
        # A human's `verified` already outranks anything this service can prove on
        # its own (identity/applicability/commencement, never a legal reading of
        # staggered commencement, repeal, or amendment history) -- overwriting it
        # with `machine_verified` would be a silent DOWNGRADE, not an upgrade, and
        # would erase that reviewer's `verified_by`/`last_verified_at` attribution
        # for no gain (`review_status` is already `approved` either way). Confirmed
        # live 2026-09-22: running this against an already human-verified document
        # did exactly that. Skip before any network fetch.
        if ((document or {}).get("metadata") or {}).get("verification_status") == "verified":
            result.update(status="skipped_already_human_verified")
            return result
        try:
            path = Path(settings.knowledge_base_dir) / policy.filename
            local = path.read_bytes()
            official, official_type = await self.fetcher(policy.official_url)
            notice, notice_type = await self.fetcher(policy.commencement_url)
            if official_type != "application/pdf" or notice_type != "application/pdf":
                raise ValueError("Official Act and commencement notice must both be PDFs.")
            local_hash = hashlib.sha256(local).hexdigest()
            official_hash = hashlib.sha256(official).hexdigest()
            if local_hash != official_hash:
                raise ValueError("Local bytes differ from the official Act PDF.")
            act_text, notice_text = normalized_text(official), normalized_text(notice)
            identity = [token in act_text for token in policy.identity_tokens]
            applicability = [token in act_text for token in policy.applicability_tokens]
            commencement = [token in notice_text for token in policy.commencement_tokens]
            if not all(identity + applicability + commencement):
                raise ValueError("One or more mandatory legal-text checks failed.")
            if not document:
                raise ValueError("No live indexed document record matches the exact official file.")
            evidence = {
                "official_url": policy.official_url, "commencement_url": policy.commencement_url,
                "official_sha256": official_hash, "local_sha256": local_hash,
                "content_sha256": hashlib.sha256(act_text.encode()).hexdigest(),
                "commencement_sha256": hashlib.sha256(notice).hexdigest(),
                "policy_version": POLICY_VERSION, "verified_at": datetime.now(UTC).isoformat(),
                "identity_checks": identity, "applicability_checks": applicability,
                "commencement_checks": commencement, "exact_byte_match": True,
                "excluded_text_patterns": list(policy.excluded_text_patterns),
            }
            publication = await self.publisher.update_machine_verified_metadata(str(document["_id"]), {
                "document_key": policy.document_key, "issuing_level": "central",
                "applicability": "all_india", "applicable_state_codes": [],
                "jurisdiction_source_type": "bare_act", "source_url": policy.official_url,
                "version_label": policy.version_label, "effective_from": policy.effective_from,
                "verification_status": "machine_verified", "metadata_provenance": "automated_official",
                "machine_verification": evidence, "section_overrides": policy.section_overrides,
            }, excluded_text_patterns=policy.excluded_text_patterns)
            result.update(status="published", **publication)
        except Exception as exc:  # noqa: BLE001 - every failed gate is an auditable hold, never a worker crash
            result["reason"] = str(exc)
            metadata = (document or {}).get("metadata") or {}
            if document and metadata.get("verification_status") == "machine_verified":
                raw = {
                    key: value for key, value in metadata.items()
                    if key not in {"review_status", "review_reasons", "last_verified_at", "verified_by"}
                }
                raw.update(verification_status="unverified", metadata_provenance="automated_official")
                await self.publisher.update_jurisdiction_metadata(str(document["_id"]), raw)
                result["quarantined"] = True
        return result


class MachineVerificationScheduler:
    """Runs `MachineVerificationService.run()` periodically -- the same
    fail-closed evidence check an operator would otherwise have to remember to
    trigger via `scripts/run_kb_machine_verification.py`. `run()` already
    rate-limits itself to once per 24h internally (`AUDIT_COLLECTION`), so a
    shorter scheduler interval just means a cheap early return, not a wasted
    fetch cycle. Mirrors `GapAutoFetchScheduler` / `OfficialSourceSyncScheduler`.
    """

    def __init__(self, service: MachineVerificationService | None = None) -> None:
        self.service = service
        self.task: Any = None

    def start(self) -> None:
        if settings.kb_machine_verification_enabled and (self.task is None or self.task.done()):
            self.service = self.service or MachineVerificationService()
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
            raise RuntimeError("Machine verification scheduler started without a service.")
        while True:
            try:
                result = await self.service.run()
                log.info(
                    "kb_machine_verification_cycle_complete",
                    status=result.get("status"), published=result.get("published"), held=result.get("held"),
                )
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the scheduler loop
                log.error("kb_machine_verification_cycle_failed", error=type(exc).__name__)
            await asyncio.sleep(max(3600, settings.kb_machine_verification_interval_hours * 3600))


machine_verification_scheduler = MachineVerificationScheduler()
