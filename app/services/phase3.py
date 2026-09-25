from __future__ import annotations

import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from uuid import uuid4

import structlog

from app.core import clock
from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.rag.kb_jurisdiction import normalize_state_code
from app.repositories.phase3 import (
    AuditLogRepository,
    BackgroundJobRepository,
    DownloadArtifactRepository,
    EvaluationRunRepository,
    FormWorkflowRepository,
    LegalSourceRepository,
    OperationalEventRepository,
    UserPreferenceRepository,
)
from app.schemas.phase3 import (
    BackgroundJobRequest,
    BackgroundJobResponse,
    EvaluationRunResponse,
    FollowUpRequest,
    FollowUpResponse,
    FormPrefillRequest,
    FormWorkflowResponse,
    LegacyReferenceResponse,
    LegalSourceMetadata,
    LegalSourceResponse,
    SourceReviewRequest,
    SupersedeSourceRequest,
    UserPreferencesRequest,
    UserPreferencesResponse,
    VerifySourceRequest,
)
from app.utils.pii import mask_pii

log = structlog.get_logger(__name__)


# Kept in sync with `app.drafting.engine.ExportFormat` -- imported lazily
# inside the job handler below (the drafting engine pulls in WeasyPrint and
# the whole export stack), so the runtime membership test needs its own copy.
_EXPORT_FORMATS = frozenset({"pdf", "docx", "txt", "rtf"})

_STALE_DAYS = 180
_LEGACY_MAP = {
    ("IPC", "302"): ("BNS", "103", "Murder provisions; subsection and facts must be checked."),
    ("IPC", "420"): ("BNS", "318(4)", "Cheating involving delivery of property; verify factual ingredients."),
    ("IPC", "498A"): ("BNS", "85", "Cruelty by husband or relative; verify current text and facts."),
    ("IPC", "354"): ("BNS", "74", "Assault/criminal force to woman; verify the precise alleged conduct."),
    ("CrPC", "154"): ("BNSS", "173", "Information in cognizable cases; verify applicable procedure."),
    ("CrPC", "41"): ("BNSS", "35", "Arrest without warrant; verify the applicable sub-provision."),
    ("CrPC", "167"): ("BNSS", "187", "Remand/investigation time limits; verify current statutory text."),
    ("CrPC", "438"): ("BNSS", "482", "Anticipatory bail; verify forum and current procedure."),
    ("CrPC", "482"): ("BNSS", "528", "High Court inherent powers; verify current text and precedent."),
}


class AuditService:
    def __init__(self) -> None:
        self.audit = AuditLogRepository()
        self.events = OperationalEventRepository()

    async def record(
        self, *, actor_user_id: str | None, action: str, resource_type: str,
        resource_id: str = "", outcome: str = "success", details: dict[str, Any] | None = None,
    ) -> None:
        safe_details = {key: mask_pii(str(value)) for key, value in (details or {}).items()}
        try:
            await self.audit.insert({
                "actor_user_id": actor_user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "outcome": outcome,
                "details": safe_details,
            })
        # Matches `event()`'s own reasoning immediately below: an audit-log
        # write failing (Mongo unavailable, e.g.) must never take down the
        # real action it is recording -- confirmed live: a real caller of
        # this method (`DocumentService.analyze`) broke outright the moment
        # this call had no Mongo connection to write through, turning an
        # audit-trail gap into a user-facing 500 for an unrelated feature.
        except Exception as exc:  # noqa: BLE001 - audit logging must never mask the real outcome it is recording
            log.warning("audit_record_write_failed", action=action, resource_type=resource_type, error=str(exc))

    async def event(self, event_type: str, *, language: str = "", details: dict[str, Any] | None = None) -> None:
        try:
            await self.events.insert({"event_type": event_type, "language": language, "details": details or {}})
        # Telemetry must never replace the original user-visible failure,
        # whatever the write failed with -- but it is logged, so a
        # permanently-broken analytics collection is visible rather than
        # presenting as "no events happened".
        except Exception as exc:  # noqa: BLE001 - telemetry must never mask the real failure
            log.warning("phase3_event_write_failed", event_type=event_type, error=str(exc))


class LegalUpdateService:
    def __init__(self) -> None:
        self.sources = LegalSourceRepository()

    def _to_response(self, item: dict[str, Any]) -> LegalSourceResponse:
        verified = item.get("last_verified_date")
        if isinstance(verified, datetime):
            verified = verified.date()
        # Security finding G3: staleness previously looked only at age and
        # `status in {"repealed", "superseded"}` -- a source reviewed today
        # and REJECTED (or merely queued `pending_review`) computed as fresh
        # ("not stale") purely because `last_verified_date` was recent, even
        # though it is explicitly not currently trustworthy law. Anything
        # short of an actual `verified` outcome is stale by definition,
        # regardless of how recently it was looked at.
        stale = (
            not verified
            or verified < clock.today() - timedelta(days=_STALE_DAYS)
            or item.get("verification_status") != "verified"
        )
        current_as_of = verified.isoformat() if verified else "Not yet verified"
        return LegalSourceResponse(
            source_id=str(item["_id"]),
            owner_user_id=item["owner_user_id"],
            document_id=item.get("document_id"),
            staging_id=item.get("staging_id"),
            stale=stale or item.get("status") in {"repealed", "superseded"},
            current_as_of=current_as_of,
            chunk_ids=list(item.get("chunk_ids") or []),
            reviewed_by=item.get("reviewed_by"),
            reviewed_at=item.get("reviewed_at"),
            review_notes=item.get("review_notes", ""),
            evidence_url=item.get("evidence_url", ""),
            # `item.get(key)` forced an explicit `None` into every metadata
            # field the stored document happened to omit, which pydantic then
            # rejected for the non-optional ones (`status`, `verification_
            # status`, ...) -- a 500 on read for a row written before a field
            # existed. Passing only the keys actually present lets each field's
            # own declared default apply instead.
            **{key: item[key] for key in LegalSourceMetadata.model_fields if key in item},
            created_at=item["created_at"],
            updated_at=item["updated_at"],
        )

    @staticmethod
    def _mongo_safe_dates(values: dict[str, Any]) -> dict[str, Any]:
        # BSON supports datetimes, not bare datetime.date values. Persist UTC
        # midnights and let the response schema expose date-only metadata.
        return {
            key: datetime.combine(value, datetime.min.time(), tzinfo=UTC) if isinstance(value, date) and not isinstance(value, datetime) else value
            for key, value in values.items()
        }

    async def create(
        self, owner_user_id: str, metadata: LegalSourceMetadata, *, document_id: str | None = None,
        staging_id: str | None = None,
    ) -> LegalSourceResponse:
        now = datetime.now(UTC)
        values = self._mongo_safe_dates(metadata.model_dump())
        # A source cannot be born verified. `create` is reachable from an admin
        # upload form, so honouring `verification_status="verified"` in the
        # request body would let the registry record a human review that never
        # happened -- exactly the claim the review workflow exists to make
        # auditable. Verification is only ever granted by `review()`, which
        # demands evidence.
        if values.get("verification_status") == "verified":
            values["verification_status"] = "pending_review"
        values.pop("superseded_by", None)
        item = {
            "_id": str(uuid4()), "owner_user_id": owner_user_id,
            **values, "document_id": document_id, "staging_id": staging_id,
            "chunk_ids": [], "reviewed_by": None, "reviewed_at": None,
            "review_notes": "", "evidence_url": "",
            "created_at": now, "updated_at": now,
        }
        await self.sources.insert(item)
        return self._to_response(item)

    async def list(self, *, stale_only: bool = False, status: str | None = None) -> list[LegalSourceResponse]:
        query: dict[str, Any] = {}
        if status:
            query["verification_status"] = status
        records = await self.sources.list_sources(query)
        results = [self._to_response(record) for record in records]
        return [item for item in results if item.stale] if stale_only else results

    async def review(
        self, source_id: str, admin_user_id: str, request: SourceReviewRequest
    ) -> LegalSourceResponse:
        """Record a human review decision, together with the evidence for it.

        `verified` is the only status that asserts a fact about the outside
        world, so it is the only one that demands an `evidence_url` and notes.
        A reviewer who cannot point at the issuing authority's own text has not
        verified anything, and letting the flip through anyway would put an
        unearned "verified" badge on an answer a user may act on.

        Security finding G3: `evidence_url`/`review_notes` are PATCH-style --
        omitted (`None`) means "leave whatever is already recorded", not
        "clear it". The `verified` requirement below is checked against the
        EFFECTIVE value (this request's, or else the source's existing one),
        so re-confirming a still-verified source without re-typing unchanged
        evidence still works, while an admin who explicitly sends `""` still
        clears it (a real, if unusual, choice to make).
        """
        item = await self.sources.find_by_id(source_id)
        if item is None:
            raise NotFoundError("Legal source not found.")
        if request.last_verified_date > clock.today():
            raise BadRequestError("A review date cannot be in the future.")
        effective_evidence_url = request.evidence_url if request.evidence_url is not None else item.get("evidence_url", "")
        effective_review_notes = request.review_notes if request.review_notes is not None else item.get("review_notes", "")
        if request.verification_status == "verified":
            if not (effective_evidence_url or "").strip():
                raise BadRequestError(
                    "Verification requires an evidence URL pointing at the issuing "
                    "authority's own publication of this source.",
                    {"field": "evidence_url"},
                )
            if not (effective_review_notes or "").strip():
                raise BadRequestError(
                    "Verification requires review notes recording what was compared.",
                    {"field": "review_notes"},
                )

        now = datetime.now(UTC)
        updates = self._mongo_safe_dates(
            request.model_dump(exclude_none=True, exclude={"evidence_url", "review_notes"})
        )
        if request.evidence_url is not None:
            updates["evidence_url"] = request.evidence_url.strip()
        if request.review_notes is not None:
            updates["review_notes"] = request.review_notes.strip()
        updates.update({
            "reviewed_by": admin_user_id,
            "reviewed_at": now,
            # Kept for records written before Phase 2 named the field.
            "verified_by": admin_user_id,
            "updated_at": now,
        })
        await self.sources.update_by_id(source_id, updates)
        item.update(updates)
        # Security finding G1: a verification/rejection state change must
        # reach every chunk this source already governs, not just the
        # `legal_sources` record itself -- otherwise retrieval keeps serving
        # (or admin review keeps showing) the governance state from whenever
        # the source was last linked, however stale that has since become.
        await self._propagate_to_chunks(item)
        return self._to_response(item)

    async def verify(
        self, source_id: str, admin_user_id: str, request: VerifySourceRequest
    ) -> LegalSourceResponse:
        """Backward-compatible wrapper over `review()` for existing callers."""
        return await self.review(
            source_id,
            admin_user_id,
            SourceReviewRequest(
                verification_status=cast(
                    "Literal['verified', 'rejected', 'pending_review']",
                    request.verification_status,
                ),
                last_verified_date=request.last_verified_date,
                evidence_url=request.evidence_url,
                review_notes=request.review_notes,
                status=request.status,
                amendment_notes=request.amendment_notes,
            ),
        )

    async def supersede(
        self, source_id: str, admin_user_id: str, request: SupersedeSourceRequest
    ) -> LegalSourceResponse:
        """Record that another registered source replaces this one in law.

        Both ends are updated so the lineage reads in either direction, and the
        replacement must already exist in the registry -- a `superseded_by`
        pointing at an id nobody registered is an amendment claim with no source
        behind it.
        """
        item = await self.sources.find_by_id(source_id)
        if item is None:
            raise NotFoundError("Legal source not found.")
        if request.superseded_by_source_id == source_id:
            raise BadRequestError("A source cannot supersede itself.")
        replacement = await self.sources.find_by_id(request.superseded_by_source_id)
        if replacement is None:
            raise NotFoundError("The superseding source is not registered.")

        now = datetime.now(UTC)
        updates: dict[str, Any] = {
            "status": "superseded",
            "superseded_by": request.superseded_by_source_id,
            "review_notes": request.review_notes.strip(),
            "reviewed_by": admin_user_id,
            "reviewed_at": now,
            "updated_at": now,
        }
        if request.effective_date is not None:
            updates["update_date"] = datetime.combine(
                request.effective_date, datetime.min.time(), tzinfo=UTC
            )
        await self.sources.update_by_id(source_id, updates)

        lineage = list(replacement.get("supersedes") or [])
        if source_id not in lineage:
            lineage.append(source_id)
            await self.sources.update_by_id(
                request.superseded_by_source_id, {"supersedes": lineage, "updated_at": now}
            )
        item.update(updates)
        # Security finding G1: see the identical call/comment in `review()`
        # -- a supersession is exactly the kind of governance change whose
        # whole point is to stop the OLD source's chunks reading as current.
        await self._propagate_to_chunks(item)
        return self._to_response(item)

    @staticmethod
    def _governance_metadata(source_id: str, item: dict[str, Any]) -> dict[str, Any]:
        """The subset of a `legal_sources` record that a linked chunk's
        `metadata.*` fields must mirror -- one definition, shared by
        `link_document` (first link) and `_propagate_to_chunks` (every
        governance change after that), so they can never drift into
        propagating different fields for the same relationship."""
        return {
            "source_id": source_id,
            "source_url": item.get("source_url", ""),
            "url": item.get("source_url", ""),
            "jurisdiction": item.get("jurisdiction", "India"),
            "act_name": item.get("act_name", ""),
            "section_number": item.get("section_number", ""),
            "source_version": item.get("source_version", ""),
            "effective_date": item.get("effective_date"),
            "amendment_status": item.get("status", "unknown"),
            "last_verified_date": item.get("last_verified_date"),
            "verification_status": item.get("verification_status", "pending_review"),
        }

    async def _propagate_to_chunks(self, item: dict[str, Any]) -> int:
        """Security finding G1: re-applies this source's CURRENT governance
        state to every chunk it already governs. Without this, `verify`/
        `review`/`supersede` updated only the `legal_sources` record --
        `link_document`'s one-time copy onto `metadata.*` (what retrieval
        and admin review actually read) simply went stale the moment the
        source's own status changed again, so a source rejected or
        superseded AFTER being linked kept presenting its chunks with
        whatever verification state happened to be true at link time.

        A no-op (not an error) for a source never linked to a document --
        `create()` and a freshly-registered supersession target both reach
        here with no `document_id` yet, and that is a normal, common state.
        """
        document_id = item.get("document_id")
        if not document_id:
            return 0
        from app.repositories.documents import EmbeddingMetadataRepository

        embeddings = EmbeddingMetadataRepository()
        metadata = self._governance_metadata(str(item.get("_id") or item.get("source_id") or ""), item)
        result = await embeddings.collection.update_many(
            {"document_id": document_id},
            {"$set": {f"metadata.{key}": value for key, value in metadata.items()}},
        )
        return result.modified_count

    async def link_document(self, source_id: str, document_id: str) -> LegalSourceResponse:
        item = await self.sources.find_by_id(source_id)
        if item is None:
            raise NotFoundError("Legal source not found.")
        from app.repositories.documents import EmbeddingMetadataRepository

        embeddings = EmbeddingMetadataRepository()
        if await embeddings.collection.count_documents({"document_id": document_id}) == 0:
            raise NotFoundError("Indexed document not found.")
        metadata = self._governance_metadata(source_id, item)
        await embeddings.collection.update_many(
            {"document_id": document_id},
            {"$set": {f"metadata.{key}": value for key, value in metadata.items()}},
        )
        # Record WHICH retrievable chunks this governance record now covers, so
        # a later verification or supersession change can be traced to the exact
        # text a user could be shown.
        chunk_ids = [
            str(row["_id"])
            async for row in embeddings.collection.find({"document_id": document_id}, {"_id": 1})
        ]
        now = datetime.now(UTC)
        await self.sources.update_by_id(
            source_id, {"document_id": document_id, "chunk_ids": chunk_ids, "updated_at": now}
        )
        item["document_id"] = document_id
        item["chunk_ids"] = chunk_ids
        item["updated_at"] = now
        return self._to_response(item)

    def map_legacy(self, code: str, section: str) -> LegacyReferenceResponse:
        normalized = section.strip().upper().replace("SECTION", "").strip()
        match = _LEGACY_MAP.get((code, normalized))
        if match is None:
            return LegacyReferenceResponse(
                legacy_reference=f"{code} Section {normalized}", current_code=None, current_section=None,
                note="No curated mapping is available. Do not infer equivalence; verify against an official concordance and current statute.",
            )
        current_code, current_section, note = match
        return LegacyReferenceResponse(
            legacy_reference=f"{code} Section {normalized}", current_code=current_code,
            current_section=current_section, note=note,
        )


class PreferenceService:
    DEFAULTS: ClassVar[dict[str, str | bool]] = {"language": "english", "explanation_mode": "simple", "preferred_document_format": "pdf", "voice_output": False}

    def __init__(self) -> None:
        self.preferences = UserPreferenceRepository()

    async def get(self, owner_user_id: str) -> UserPreferencesResponse:
        record = await self.preferences.get_for_owner(owner_user_id)
        return UserPreferencesResponse(**{**self.DEFAULTS, **(record or {})})

    async def update(self, owner_user_id: str, request: UserPreferencesRequest) -> UserPreferencesResponse:
        values = request.model_dump(exclude_none=True)
        if "profile_state_code" in values:
            code = normalize_state_code(values["profile_state_code"])
            if code is None:
                raise BadRequestError(f"Unknown State/UT code or name: {values['profile_state_code']!r}.")
            values["profile_state_code"] = code
        await self.preferences.upsert_for_owner(owner_user_id, values)
        return await self.get(owner_user_id)

    async def delete(self, owner_user_id: str) -> None:
        await self.preferences.collection.delete_one({"owner_user_id": owner_user_id})


_FOLLOWUPS = {
    "cyber_fraud": [
        ("transaction_time", "When did the transaction happen?"),
        ("transaction_id", "What is the UTR or transaction reference?"),
        ("platform", "Which app or platform was used?"),
        ("bank", "Which bank or payment provider is involved?"),
        ("screenshots", "Do you have transaction screenshots or receipts?"),
        ("report_status", "Have you reported it to the bank, 1930, or cybercrime.gov.in?"),
    ],
    "police_complaint": [
        ("location", "Where did the incident happen?"),
        ("accused", "Is the accused known or unknown?"),
        ("witnesses", "Were there any witnesses?"),
        ("evidence", "What evidence is currently available?"),
    ],
}

_LOCALIZED_FOLLOWUPS = {
    "hindi": {
        "transaction_time": "लेन-देन कब हुआ था?",
        "transaction_id": "UTR या लेन-देन संदर्भ संख्या क्या है?",
        "platform": "कौन-सा ऐप या प्लेटफ़ॉर्म इस्तेमाल हुआ था?",
        "bank": "कौन-सा बैंक या भुगतान प्रदाता शामिल है?",
        "screenshots": "क्या आपके पास लेन-देन के स्क्रीनशॉट या रसीद हैं?",
        "report_status": "क्या आपने बैंक, 1930 या cybercrime.gov.in पर रिपोर्ट की है?",
        "location": "घटना कहाँ हुई थी?",
        "accused": "आरोपी ज्ञात है या अज्ञात?",
        "witnesses": "क्या कोई गवाह था?",
        "evidence": "अभी आपके पास कौन-से सबूत उपलब्ध हैं?",
    },
    "hinglish": {
        "transaction_time": "Transaction kab hua tha?",
        "transaction_id": "UTR ya transaction reference number kya hai?",
        "platform": "Kaunsa app ya platform use hua tha?",
        "bank": "Kaunsa bank ya payment provider involved hai?",
        "screenshots": "Kya transaction ke screenshots ya receipt available hain?",
        "report_status": "Kya aapne bank, 1930, ya cybercrime.gov.in par report kiya hai?",
        "location": "Incident kahan hua tha?",
        "accused": "Accused known hai ya unknown?",
        "witnesses": "Kya koi witness tha?",
        "evidence": "Abhi kaunsa evidence available hai?",
    },
}

_FIELD_CUES = {
    "transaction_time": ("time", "date", "when", "बजे", "तारीख"),
    "transaction_id": ("utr", "transaction id", "reference", "txn"),
    "platform": ("upi", "phonepe", "gpay", "paytm", "platform", "app"),
    "bank": ("bank", "sbi", "hdfc", "icici", "axis", "बैंक"),
    "screenshots": ("screenshot", "receipt", "image", "स्क्रीनशॉट"),
    "report_status": ("1930", "cybercrime.gov.in", "reported", "complaint number"),
    "location": ("location", "address", "police station", "place", "जगह"),
    "accused": ("accused", "unknown person", "known person", "आरोपी"),
    "witnesses": ("witness", "गवाह"),
    "evidence": ("evidence", "video", "photo", "recording", "सबूत"),
}


def next_follow_up(request: FollowUpRequest) -> FollowUpResponse:
    corpus = " ".join(
        [str(message.get("content", "")) for message in request.messages]
        + request.document_texts
    ).casefold()
    known = {key for key, value in request.known_fields.items() if value not in (None, "", [], {})}
    for field, cues in _FIELD_CUES.items():
        if any(cue.casefold() in corpus for cue in cues):
            known.add(field)
    # Conversational reports commonly provide a bare clock time or a relative
    # date without saying "time"/"date". Treat those as supplied facts, but do
    # not attempt to normalize them here; the review workflow still shows the
    # user's exact wording for confirmation.
    if re.search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s*(?:am|pm))?\b", corpus) or re.search(
        r"\b(?:today|yesterday|tonight|कल|आज)\b", corpus
    ):
        known.add("transaction_time")
    unanswered = [field for field, _ in _FOLLOWUPS[request.workflow] if field not in known]
    selectable = [(field, question) for field, question in _FOLLOWUPS[request.workflow] if field in unanswered and field not in request.asked_fields]
    if not selectable:
        return FollowUpResponse(field=None, question=None, complete=not unanswered, remaining_fields=unanswered)
    field, question = selectable[0]
    question = _LOCALIZED_FOLLOWUPS.get(request.language, {}).get(field, question)
    return FollowUpResponse(field=field, question=question, complete=False, remaining_fields=unanswered)


_FORM_REQUIRED = {
    "police_complaint": ("complainant_name", "complainant_address", "incident_date", "incident_location", "facts", "relief_requested"),
    "cybercrime_complaint": ("complainant_name", "mobile", "incident_date", "amount", "transaction_id", "bank", "platform", "facts"),
    "rti": ("applicant_name", "applicant_address", "public_authority", "information_requested"),
    "consumer_complaint": ("complainant_name", "complainant_address", "respondent_name", "transaction_date", "amount", "deficiency", "relief_requested"),
}


class FormWorkflowService:
    def __init__(self) -> None:
        self.forms = FormWorkflowRepository()

    def _response(self, item: dict[str, Any]) -> FormWorkflowResponse:
        missing = [key for key in _FORM_REQUIRED[item["form_type"]] if not item.get("fields", {}).get(key)]
        return FormWorkflowResponse(
            workflow_id=item["_id"], form_type=item["form_type"], fields=item.get("fields", {}),
            missing_fields=missing, confirmed=item.get("confirmed", False),
            warning="Review every field and annexure before export. This assistant does not submit anything to an external authority.",
        )

    async def create(self, owner_user_id: str, request: FormPrefillRequest) -> FormWorkflowResponse:
        now = datetime.now(UTC)
        item = {
            "_id": str(uuid4()), "owner_user_id": owner_user_id, "form_type": request.form_type,
            "case_id": request.case_id, "fields": request.facts, "evidence": request.evidence,
            "language": request.language, "confirmed": False, "created_at": now, "updated_at": now,
        }
        await self.forms.insert(item)
        return self._response(item)

    async def list(self, owner_user_id: str) -> list[FormWorkflowResponse]:
        return [self._response(item) for item in await self.forms.list_for_owner(owner_user_id)]

    async def get(self, owner_user_id: str, workflow_id: str) -> FormWorkflowResponse:
        return self._response(await self._owned(owner_user_id, workflow_id))

    async def _owned(self, owner_user_id: str, workflow_id: str) -> dict[str, Any]:
        item = await self.forms.find_by_id(workflow_id)
        if item is None:
            raise NotFoundError("Form workflow not found.")
        if item.get("owner_user_id") != owner_user_id:
            raise ForbiddenError("You do not have access to this form workflow.")
        return item

    async def update(self, owner_user_id: str, workflow_id: str, fields: dict[str, Any]) -> FormWorkflowResponse:
        item = await self._owned(owner_user_id, workflow_id)
        merged = {**item.get("fields", {}), **fields}
        await self.forms.update_by_id(workflow_id, {"fields": merged, "confirmed": False})
        item.update({"fields": merged, "confirmed": False})
        return self._response(item)

    async def confirm(self, owner_user_id: str, workflow_id: str, confirmation_text: str) -> FormWorkflowResponse:
        item = await self._owned(owner_user_id, workflow_id)
        response = self._response(item)
        if response.missing_fields:
            raise BadRequestError("Complete all required fields before confirmation.", {"missing_fields": response.missing_fields})
        if confirmation_text.strip().upper() != "I CONFIRM THE REVIEWED FACTS":
            raise BadRequestError("Type 'I CONFIRM THE REVIEWED FACTS' to confirm the review screen.")
        await self.forms.update_by_id(workflow_id, {"confirmed": True, "confirmed_at": datetime.now(UTC)})
        item["confirmed"] = True
        return self._response(item)


class BackgroundJobService:
    def __init__(self) -> None:
        self.jobs = BackgroundJobRepository()
        self.downloads = DownloadArtifactRepository()

    async def create(self, owner_user_id: str, request: BackgroundJobRequest) -> BackgroundJobResponse:
        now = datetime.now(UTC)
        item = {
            "_id": str(uuid4()), "owner_user_id": owner_user_id, "job_type": request.job_type,
            "payload": request.payload, "case_id": request.case_id, "status": "queued", "progress": 0,
            "error": None, "result": {}, "attempts": 0, "created_at": now, "updated_at": now,
        }
        await self.jobs.insert(item)
        return self._response(item)

    async def get(self, owner_user_id: str, job_id: str) -> BackgroundJobResponse:
        item = await self.jobs.find_by_id(job_id)
        if item is None:
            raise NotFoundError("Background job not found.")
        if item.get("owner_user_id") != owner_user_id:
            raise ForbiddenError("You do not have access to this background job.")
        return self._response(item)

    async def list(self, owner_user_id: str) -> list[BackgroundJobResponse]:
        return [self._response(item) for item in await self.jobs.list_for_owner(owner_user_id)]

    async def retry(self, owner_user_id: str, job_id: str) -> BackgroundJobResponse:
        item = await self.jobs.find_by_id(job_id)
        if item is None:
            raise NotFoundError("Background job not found.")
        if item.get("owner_user_id") != owner_user_id:
            raise ForbiddenError("You do not have access to this background job.")
        if item.get("status") != "failed":
            raise BadRequestError("Only failed jobs can be retried.")
        now = datetime.now(UTC)
        await self.jobs.update_by_id(job_id, {
            "status": "queued", "progress": 0, "error": None, "worker_id": None,
            "started_at": None, "completed_at": None, "retried_at": now,
        })
        item.update({"status": "queued", "progress": 0, "error": None, "updated_at": now})
        return self._response(item)

    def _response(self, item: dict[str, Any]) -> BackgroundJobResponse:
        return BackgroundJobResponse(
            job_id=item["_id"], job_type=item["job_type"], status=item["status"],
            progress=item.get("progress", 0), error=item.get("error"), result=item.get("result", {}),
            created_at=item["created_at"], updated_at=item["updated_at"],
        )

    async def process_next(self, worker_id: str) -> bool:
        job = await self.jobs.claim_next(worker_id)
        if job is None:
            return False
        try:
            result = await self._execute(job)
            await self.jobs.update_by_id(job["_id"], {"status": "complete", "progress": 100, "result": result, "completed_at": datetime.now(UTC)})
        except Exception as exc:  # noqa: BLE001 - a background job's failure is RECORDED on the job document; the worker loop must keep claiming work
            await self.jobs.update_by_id(job["_id"], {"status": "failed", "error": str(exc)[:1000], "completed_at": datetime.now(UTC)})
        return True

    async def _execute(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job.get("payload", {})
        job_type = job["job_type"]
        if job_type == "export":
            from app.drafting.engine import ExportFormat, LegalDraftEngine
            from app.drafting.export import ExportOptions

            engine = LegalDraftEngine()
            draft = await engine.drafts.find_by_id(str(payload.get("draft_id", "")))
            if draft is None or draft.get("user_id") != job["owner_user_id"]:
                raise ForbiddenError("Draft is missing or does not belong to the job owner.")
            fmt = str(payload.get("format", "pdf"))
            if fmt not in _EXPORT_FORMATS:
                raise BadRequestError(
                    f"Unsupported export format: {fmt}",
                    {"supported": sorted(_EXPORT_FORMATS)},
                )
            fmt = cast(ExportFormat, fmt)
            path = await engine.export(draft["_id"], fmt, ExportOptions(watermark=bool(payload.get("watermark", False))))
            artifact_id = await self.downloads.insert({
                "owner_user_id": job["owner_user_id"], "job_id": job["_id"], "draft_id": draft["_id"],
                "filename": f"{draft['_id']}.{fmt}", "format": fmt, "path": str(path),
            })
            return {"artifact_id": artifact_id, "filename": f"{draft['_id']}.{fmt}"}
        if job_type == "document_analysis":
            from app.schemas.document import DocumentAnalysisRequest
            from app.services.document_service import DocumentService

            analysis = await DocumentService().analyze(
                DocumentAnalysisRequest(document_id=payload.get("document_id"), session_id=payload.get("session_id")),
                authenticated_user_id=job["owner_user_id"],
            )
            return {"analysis": analysis.model_dump(mode="json")}
        if job_type in {"ingestion", "ocr", "indexing"}:
            # These jobs are produced/consumed by the existing staged KB and
            # document pipelines. The durable ledger records the hand-off;
            # raw filesystem paths are deliberately never accepted here.
            if not payload.get("document_id") and not payload.get("staging_id"):
                raise BadRequestError(f"{job_type} requires document_id or staging_id.")
            return {"handoff": "specialized_pipeline", "reference": payload.get("document_id") or payload.get("staging_id")}
        raise BadRequestError(f"Unsupported job type: {job_type}")


class EvaluationService:
    DEFAULT_BENCHMARK = (
        Path(__file__).resolve().parent.parent.parent
        / "tests"
        / "benchmarks"
        / "phase3_conversations_v1.json"
    )

    def __init__(self) -> None:
        self.runs = EvaluationRunRepository()

    async def run(self, benchmark_path: str | None = None) -> EvaluationRunResponse:
        # Importing the evaluator registers ChatOps workflows, several of
        # which use services from this module. Keep that import at the call
        # boundary so importing Phase-3 services cannot create a cycle.
        from app.chatops.evaluation import evaluate_conversations

        started = time.perf_counter()
        benchmark_root = self.DEFAULT_BENCHMARK.parent.resolve()
        path = (Path(benchmark_path) if benchmark_path else self.DEFAULT_BENCHMARK).resolve()
        if benchmark_root != path.parent:
            raise BadRequestError("Benchmark files must be selected from the curated tests/benchmarks directory.")
        if not path.is_file():
            raise BadRequestError(f"Benchmark does not exist: {path}")
        try:
            evaluation = evaluate_conversations(path)
        except (OSError, ValueError) as exc:
            raise BadRequestError(f"Invalid conversation benchmark: {exc}") from exc
        scores = {
            "routing_accuracy": evaluation.routing_accuracy,
            "language_accuracy": evaluation.language_accuracy,
            "ordinary_question_safety": evaluation.ordinary_question_safety,
        }
        failures = [{"detail": failure} for failure in evaluation.failures]
        latency_ms = (time.perf_counter() - started) * 1000
        run_id = await self.runs.insert({
            "status": "complete", "benchmark": str(path), "scores": scores,
            "case_count": evaluation.cases, "failures": failures, "latency_ms": latency_ms,
        })
        return EvaluationRunResponse(
            run_id=run_id, status="complete", scores=scores, cases=evaluation.cases,
            failures=failures, latency_ms=latency_ms,
        )

    def _to_response(self, item: dict[str, Any]) -> EvaluationRunResponse:
        return EvaluationRunResponse(
            run_id=str(item["_id"]), status=item.get("status", "complete"),
            scores=item.get("scores", {}), cases=item.get("case_count", 0),
            failures=item.get("failures", []), latency_ms=item.get("latency_ms", 0.0),
        )

    async def list_runs(self, limit: int = 50) -> list[EvaluationRunResponse]:
        """Security/correctness finding G6: see `EvaluationRunRepository.
        list_recent`'s docstring -- historical runs were durably stored but
        had no read path at all."""
        records = await self.runs.list_recent(limit)
        return [self._to_response(record) for record in records]

    async def get_run(self, run_id: str) -> EvaluationRunResponse:
        record = await self.runs.find_by_id(run_id)
        if record is None:
            raise NotFoundError(f"Evaluation run '{run_id}' not found.")
        return self._to_response(record)
