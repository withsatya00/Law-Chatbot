import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings
from app.core.constants import ALLOWED_UPLOAD_EXTENSIONS
from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.drafting.localized_dates import MONTH_WORD_TO_NUMBER, parse_localized_date
from app.entity_extraction.extractor import EntityExtractor
from app.llm.base import ChatMessage, LLMProvider
from app.llm.factory import LLMFactory
from app.llm.prompts import prompt_registry
from app.memory.store import ConversationMemoryStore
from app.rag.loader import DocumentLoader
from app.rag.pipeline import IndexingPipeline
from app.recommendation.engine import LawyerRecommendationEngine
from app.repositories.documents import DocumentRepository, EmbeddingMetadataRepository
from app.schemas.common import LawyerRecommendation
from app.schemas.document import (
    AnalyzedClause,
    DocumentAnalysisRequest,
    DocumentAnalysisResponse,
    DocumentTimelineEvent,
    UploadResponse,
)
from app.services.phase3 import AuditService
from app.utils.malware_scan import ClamAVScanner, MalwareScanner
from app.utils.prompt_security import PromptInjectionScanner
from app.utils.upload_storage import write_upload

# Risk/clause signal words the naive keyword scan below looks for. Kept
# intentionally broader than the original ["penalty", "breach", "default",
# "liability", "termination"] list, which missed extremely common risk
# phrasing in real contracts (a no-refund-under-any-circumstances deposit
# clause, a landlord's unilateral entry-without-notice right, forfeiture on
# early exit) -- confirmed directly against a test document containing
# exactly those clauses, which the original list caught none of.
_CLAUSE_TERMS = ["termination", "payment", "penalty", "liability", "jurisdiction", "notice", "deposit", "rent"]
_RISK_TERMS = [
    "penalty",
    "breach",
    "default",
    "liability",
    "liable",
    "termination",
    "forfeit",
    "waive",
    "waives",
    "without notice",
    "without prior notice",
    "sole discretion",
    "non-refundable",
    "not be liable",
    "indemnify",
    "disclaim",
]


def ensure_document_access(
    metadata: dict[str, Any], authenticated_user_id: str | None, session_id: str | None
) -> None:
    """Part 46 "Authenticated User Ownership": global -> anyone;
    `owner_user_id` set -> only that exact authenticated user, from any
    session; `owner_session_id`-only set (Part 45-era, or an anonymous
    upload) and no `owner_user_id` -> only that exact session -- mirrors
    the same three-way visibility logic `ChatService._prepare_rag_context`
    applies via the retrieval filter, just as a direct check here since
    this is a single-document lookup, not a retrieval scan.

    Module-level so the document-review and comparison workflows can call
    exactly this rule rather than approximating it: "may this person see
    this document" must have one answer in this codebase.
    """
    owner_user_id = metadata.get("owner_user_id")
    owner_session_id = metadata.get("owner_session_id")
    if not owner_user_id and not owner_session_id:
        return
    if owner_user_id:
        if authenticated_user_id and authenticated_user_id == owner_user_id:
            return
        raise ForbiddenError("You do not have access to this document.")
    if session_id and session_id == owner_session_id:
        return
    raise ForbiddenError("You do not have access to this document.")


class DocumentService:
    def __init__(self, llm: LLMProvider | None = None, scanner: MalwareScanner | None = None) -> None:
        self.pipeline = IndexingPipeline(loader=DocumentLoader(handwriting=True))
        self.documents = DocumentRepository()
        self.embeddings = EmbeddingMetadataRepository()
        self.recommendations = LawyerRecommendationEngine()
        self.memory = ConversationMemoryStore()
        self.llm = llm or LLMFactory.create()
        self.entities = EntityExtractor()
        self.scanner = scanner or ClamAVScanner()
        self.prompt_scanner = PromptInjectionScanner()

    async def delete_owned(
        self, document_id: str, authenticated_user_id: str | None = None, session_id: str | None = None,
    ) -> int:
        """Permanently removes one uploaded document: every indexed chunk
        (Mongo and the BM25 mirror, via `delete_version_chunks` -- the SAME
        method `IndexingPipeline`'s own reindex-cleanup path already uses,
        rather than a second, independently-written deletion query that
        could drift from it) and the document record itself. Returns the
        number of chunks removed.

        PDF Q&A acceptance pass (2026-09-12): there was previously no way
        for a user to delete an uploaded document at all -- `DELETE
        /session`/`DELETE /me/data` (`app/services/user_data.py`) erase
        conversation memory and chat history, never indexed document
        content, so a "deleted" document's text and page evidence stayed
        fully queryable indefinitely.

        Ownership-checked exactly like every other document-reading path
        (`ensure_document_access`), against the SAME `owner_user_id`/
        `owner_session_id` fields `IndexingPipeline.index_file` writes onto
        the document record itself (not just its chunks) -- `NotFoundError`
        and `ForbiddenError` are deliberately distinguished, the same
        convention `document_insight.load_document` already uses.
        """
        document = await self.documents.find_by_id(document_id)
        if document is None:
            raise NotFoundError("That document is not available.")
        self._ensure_document_access(document, authenticated_user_id, session_id)
        deleted_chunks = await self.pipeline.vector_store.delete_version_chunks(document_id)
        await self.documents.delete_by_id(document_id)
        await AuditService().record(
            actor_user_id=authenticated_user_id, action="document_deleted",
            resource_type="uploaded_document", resource_id=document_id,
            details={"session_id": session_id or "", "chunks_deleted": str(deleted_chunks)},
        )
        return deleted_chunks

    async def list_owned(
        self, user_id: str | None = None, session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List only the caller's private uploads; never enumerate shared KB files."""
        if user_id:
            query: dict[str, Any] = {"owner_user_id": user_id}
        elif session_id:
            await self.memory.check_access(session_id, None)
            query = {"owner_user_id": None, "owner_session_id": session_id}
        else:
            return []
        projection = {
            "_id": 1, "filename": 1, "original_filename": 1, "detected_language": 1, "chunk_count": 1,
            "index_status": 1, "created_at": 1, "updated_at": 1,
        }
        return [
            {
                "document_id": str(item["_id"]),
                # Security/correctness finding K2: prefer the caller's own
                # uploaded filename over the internal storage-key-derived
                # one -- `original_filename` is absent on documents indexed
                # before this field existed, so this stays a safe,
                # backward-compatible fallback rather than a hard migration.
                "filename": item.get("original_filename") or item.get("filename", "document"),
                "detected_language": item.get("detected_language"),
                "chunks_indexed": int(item.get("chunk_count") or 0),
                "status": item.get("index_status", "unknown"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            }
            async for item in self.documents.collection.find(query, projection).sort("created_at", -1)
        ]

    async def upload_and_index(
        self, file: UploadFile, session_id: str | None = None, user_id: str | None = None,
    ) -> UploadResponse:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            raise BadRequestError(f"Unsupported upload type: {suffix}")
        settings.upload_storage_dir.mkdir(parents=True, exist_ok=True)
        destination = settings.upload_storage_dir / f"{uuid4()}{suffix}"
        await write_upload(file, destination)
        try:
            await self.scanner.scan(destination)
        except ValueError as exc:
            # Never index or retain a file that failed the scan -- mirrors
            # the admin KB-ingestion path (`kb_automation.ClamAVScanner`
            # usage), which previously was the only upload path with any
            # malware scanning; `/upload` had extension/size checks only.
            destination.unlink(missing_ok=True)
            raise BadRequestError(f"Upload rejected: {exc}") from exc
        # Part 45 "Per-User Document Isolation": no `session_id` (any caller
        # that isn't the chat UI -- scripts, admin tooling, tests) means this
        # document has no owner and stays globally visible, matching every
        # document's behavior before this field existed. Not required/
        # enforced -- see the plan's rationale for keeping it optional.
        # Part 46 "Authenticated User Ownership": `user_id` is the
        # JWT-verified identity from the route's `Depends(get_current_user_id)`
        # (`None` for an anonymous upload) -- authoritative when present,
        # never a client-supplied value.
        document_id, language, chunks = await self.pipeline.index_file(
            destination, owner_session_id=session_id, owner_user_id=user_id,
            original_filename=file.filename,
        )
        # Part 51 "Uploaded Document Conversation Context": remembered only
        # after indexing has genuinely succeeded (every failure path above --
        # bad extension, over the size limit, or an exception from
        # `index_file` itself -- returns/raises before this line), and only
        # when there's a conversation session to remember it FOR.
        if session_id:
            # `uploaded_documents` is the LIST of everything this conversation
            # has uploaded, alongside the single most-recent id. The list is
            # what makes "compare these two agreements" and "which document
            # did you mean?" possible at all -- with only the latest id, a
            # second upload silently replaced the first and the user had no
            # way to refer back to it except by re-uploading.
            existing = await self.memory.load(session_id)
            documents = [
                item for item in (existing.get("uploaded_documents") or [])
                if isinstance(item, dict) and item.get("document_id") != document_id
            ]
            documents.append({
                "document_id": document_id,
                "filename": file.filename or destination.name,
                "language": language,
                "uploaded_at": datetime.now(UTC).isoformat(),
            })
            await self.memory.update(
                session_id,
                last_uploaded_document_id=document_id,
                # Bounded: a conversation that uploads dozens of files still
                # only needs the recent ones to be referable by name.
                uploaded_documents=documents[-10:],
            )
        metadata = chunks[0].metadata if chunks else {}
        warnings: list[str] = []
        if metadata.get("ocr_degraded"):
            # Previously this was silent: a scanned PDF whose OCR was
            # unavailable/disabled or added nothing over its near-empty
            # embedded text layer got indexed anyway, and the only trace was
            # a server-side log line -- the user had no way to know their
            # document was effectively ingested blank and would get poor or
            # empty answers back. See `DocumentLoader._load_pdf`.
            reason = metadata.get("ocr_degraded_reason")
            detail_by_reason = {
                "ocr_disabled": "OCR is currently disabled on this server, so no text could be extracted from it.",
                "ocr_unavailable": "the OCR engine needed to read scans is unavailable right now.",
                "ocr_no_improvement": "OCR ran but could not extract meaningful text from the scan quality.",
                "ocr_low_confidence": "OCR ran but had low confidence reading it -- it may be in a script this "
                "server's OCR isn't tuned for, or the scan quality is poor.",
            }
            fallback_detail = "it could not be read reliably."
            detail = detail_by_reason.get(reason, fallback_detail) if isinstance(reason, str) else fallback_detail
            warnings.append(
                f"'{file.filename or destination.name}' looks like a scanned document, but {detail} "
                "Answers based on this document may be incomplete or missing -- consider uploading a "
                "clearer scan or a text-based version."
            )
        return UploadResponse(
            document_id=document_id,
            filename=file.filename or destination.name,
            status="indexed",
            chunks_indexed=len(chunks),
            detected_language=language,
            metadata=metadata,
            warnings=warnings,
        )

    async def analyze(
        self, request: DocumentAnalysisRequest, authenticated_user_id: str | None = None,
    ) -> DocumentAnalysisResponse:
        text = request.text or ""
        if request.document_id:
            # Bug fix: this previously set `text` to `str(document["metadata"])`
            # -- a repr of the metadata dict, not the document's actual
            # content -- so every analysis (summary, risks, clauses) ran over
            # a metadata dump instead of the document. The real text lives in
            # the chunks this document was split into at indexing time.
            cursor = self.embeddings.collection.find({"document_id": request.document_id}, {"text": 1, "metadata": 1})
            chunks = [chunk async for chunk in cursor]
            # Part 46 "Authenticated User Ownership": closes the gap flagged
            # as a remaining risk in Part 45 -- this lookup previously had no
            # ownership check at all, unlike the RAG-retrieval path. Ownership
            # fields are document-wide, so the first chunk's metadata is
            # representative of the whole document.
            if chunks:
                self._ensure_document_access(chunks[0].get("metadata") or {}, authenticated_user_id, request.session_id)
                # A stored document being read back is the actual
                # access event worth a trail (a raw pasted `text` above
                # is never persisted, so there is nothing to have
                # "accessed") -- previously nothing recorded who looked
                # at which uploaded document, unlike admin KB actions
                # and (now) draft export.
                await AuditService().record(
                    actor_user_id=authenticated_user_id, action="document_accessed",
                    resource_type="uploaded_document", resource_id=request.document_id,
                    details={"session_id": request.session_id or ""},
                )
            text = "\n\n".join(chunk.get("text", "") for chunk in chunks)
        if not text.strip():
            raise BadRequestError("Provide document_id or text for analysis.")
        # An uploaded document's text goes straight into an LLM prompt below
        # (`_llm_analysis`) -- previously the only unguarded LLM-input path;
        # chat (`ChatService`) and drafting (`DraftConversationEngine`,
        # `LegalDraftEngine`) already scan their inputs with this same
        # scanner. A hostile document is a lower-friction injection vector
        # than a hostile chat message since the user never sees its content.
        risky, findings = self.prompt_scanner.scan(text)
        if risky:
            raise BadRequestError("The document contains unsafe prompt-injection instructions.", {"findings": findings})
        entity_result = await self.entities.extract(text, request.language)
        timeline = self._extract_timeline(text, request.language)
        recommendation = await self.recommendations.recommend("Document Analysis", "General Law")
        llm_analysis = await self._llm_analysis(text, request, recommendation)
        if llm_analysis is None:
            return self._fallback_analysis(text, request, recommendation, entity_result.entities, timeline)
        clauses = llm_analysis.pop("analyzed_clauses", [])
        return DocumentAnalysisResponse(
            executive_summary=llm_analysis["executive_summary"],
            legal_summary=llm_analysis["legal_summary"],
            important_clauses=llm_analysis["important_clauses"],
            important_dates=llm_analysis["important_dates"],
            important_names=llm_analysis["important_names"],
            important_sections=llm_analysis["important_sections"],
            key_risks=llm_analysis["key_risks"],
            action_items=llm_analysis["action_items"],
            missing_information=llm_analysis["missing_information"],
            structured_data={**llm_analysis["structured_data"], "analysis_type": request.analysis_type, "entities": entity_result.entities},
            sources=[],
            recommended_lawyer=recommendation,
            confidence=llm_analysis["confidence"],
            analyzed_clauses=clauses,
            timeline=timeline or llm_analysis.get("timeline", []),
        )

    async def _llm_analysis(
        self, text: str, request: DocumentAnalysisRequest, recommendation: LawyerRecommendation,
    ) -> dict[str, Any] | None:
        bounded_text = text[: settings.document_analysis_max_chars]
        try:
            prompt = prompt_registry.render(
                "document_analysis_prompt",
                document_type=request.analysis_type,
                language=request.language or "the document's language",
                document_text=bounded_text,
                context="None provided",
            )
            response = await self.llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.0)
            if response.error or not response.content.strip():
                return None
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return None
            required_lists = (
                "important_clauses", "important_dates", "important_names", "important_sections",
                "key_risks", "action_items", "missing_information",
            )
            if any(not isinstance(parsed.get(field), list) for field in required_lists):
                return None
            if not isinstance(parsed.get("structured_data"), dict):
                return None
            parsed["analyzed_clauses"] = [AnalyzedClause.model_validate(clause).model_dump() for clause in parsed.get("analyzed_clauses", [])]
            parsed["timeline"] = [DocumentTimelineEvent.model_validate(event).model_dump() for event in parsed.get("timeline", [])]
            validated = DocumentAnalysisResponse(
                **parsed,
                sources=[],
                recommended_lawyer=recommendation,
            )
            return validated.model_dump(exclude={"sources", "recommended_lawyer"})
        except Exception:  # noqa: BLE001 - analysis must fall back on provider outages too
            return None

    def _fallback_analysis(
        self, text: str, request: DocumentAnalysisRequest, recommendation: LawyerRecommendation,
        entities: dict[str, Any], timeline: list[DocumentTimelineEvent],
    ) -> DocumentAnalysisResponse:
        return DocumentAnalysisResponse(
            executive_summary=text[:600],
            legal_summary="The document was parsed and should be reviewed with the cited source context before action.",
            important_clauses=self._find_lines(text, _CLAUSE_TERMS),
            important_dates=self._find_dates(text),
            important_names=entities.get("person_name", []),
            important_sections=self._find_lines(text, ["section", "article", "clause"]),
            key_risks=self._find_lines(text, _RISK_TERMS),
            action_items=["Verify document authenticity.", "Preserve the original copy.", "Consult a qualified advocate."],
            missing_information=["Jurisdiction and execution details should be verified if absent."],
            structured_data={"analysis_type": request.analysis_type, "analysis_source": "deterministic_fallback", "entities": entities},
            sources=[], recommended_lawyer=recommendation, confidence=0.64,
            timeline=timeline,
        )

    def _extract_timeline(self, text: str, language: str | None) -> list[DocumentTimelineEvent]:
        date_pattern = re.compile(
            r"(?:\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+(?:[A-Za-z]+|"
            + "|".join(re.escape(month) for month in MONTH_WORD_TO_NUMBER)
            + r")\s+\d{4}\b)"
        )
        events: list[tuple[date | None, int, DocumentTimelineEvent]] = []
        for match in date_pattern.finditer(text):
            raw_date = match.group(0)
            parsed_date = self._parse_timeline_date(raw_date)
            start = text.rfind("\n", 0, match.start()) + 1
            end_candidates = [value for value in (text.find("\n", match.end()), text.find(".", match.end())) if value >= 0]
            end = min(end_candidates) + 1 if end_candidates else len(text)
            source = text[start:end].strip()
            description = re.sub(r"\s+", " ", source).strip(" .") or raw_date
            events.append((parsed_date, match.start(), DocumentTimelineEvent(
                date=parsed_date.isoformat() if parsed_date else raw_date,
                event_description=description,
                source_text=source,
            )))
        events.sort(key=lambda item: (item[0] is None, item[0] or date.max, item[1]))
        return [event for _, _, event in events[:20]]

    def _parse_timeline_date(self, value: str) -> date | None:
        for pattern in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(value, pattern).replace(tzinfo=UTC).date()
            except ValueError:
                continue
        return parse_localized_date(value)

    def _ensure_document_access(
        self, metadata: dict[str, Any], authenticated_user_id: str | None, session_id: str | None,
    ) -> None:
        """The document ownership rule. Delegates to the module-level
        function so every caller -- this service, the chat workflows that
        review and compare documents -- enforces one implementation."""
        ensure_document_access(metadata, authenticated_user_id, session_id)

    def _find_lines(self, text: str, terms: list[str]) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return [line for line in lines if any(term in line.lower() for term in terms)][:10]

    def _find_dates(self, text: str) -> list[str]:
        import re

        return re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)[:20]
