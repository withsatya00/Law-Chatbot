import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from app.core.config import settings
from app.core.exceptions import BadRequestError
from app.language.detector import LanguageDetector
from app.mlops.registry import RuntimeVersionRegistry
from app.rag.chunker import SectionAwareChunker
from app.rag.embeddings import EmbeddingProvider
from app.rag.kb_jurisdiction import apply_to_chunk
from app.rag.loader import DocumentLoader
from app.rag.metadata import MetadataExtractor
from app.rag.quality import DocumentQualityChecker
from app.rag.types import DocumentChunk, LoadedDocument
from app.rag.vector_store import MongoVectorStore, VectorStore
from app.repositories.documents import DocumentRepository
from app.repositories.versioning import DocumentVersionRepository

log = structlog.get_logger(__name__)


class IndexingPipeline:
    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: SectionAwareChunker | None = None,
        embeddings: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or SectionAwareChunker()
        self.embeddings = embeddings or EmbeddingProvider()
        self.vector_store = vector_store or MongoVectorStore()
        self.metadata_extractor = MetadataExtractor()
        self.language_detector = LanguageDetector()
        self.documents = DocumentRepository()
        self.versions = DocumentVersionRepository()
        self.quality_checker = DocumentQualityChecker()
        self.runtime_versions = RuntimeVersionRegistry()

    async def index_file(
        self,
        path: Path,
        owner_session_id: str | None = None,
        owner_user_id: str | None = None,
        jurisdiction_metadata: dict[str, Any] | None = None,
        original_filename: str | None = None,
    ) -> tuple[str, str, list[DocumentChunk]]:
        """`original_filename` (security/correctness finding K2): `path`'s
        own name is a server-generated storage key (`DocumentService.
        upload_and_index` writes every private upload to
        `f"{uuid4()}{suffix}"`, deliberately never the caller's own
        filename, to keep the actual filesystem write injection-safe), and
        `DocumentLoader.load` stamps `document.filename` from exactly that
        path name -- so `source_document`/`document.filename` is that same
        opaque storage key for every private upload, and a caller listing
        their own documents (`DocumentService.list_owned`) saw a raw UUID
        where a filename belongs (confirmed live: `document_versions` has
        real rows with `source_document` values like
        "62f33814-076e-4750-833f-2eb3aad43e00.pdf" instead of the file the
        user actually uploaded). `document.filename`/`source_document`
        themselves are left alone here -- they are the stable identity
        version lineage/dedup already key on, and two different accounts
        uploading a same-named "agreement.pdf" must never collide just
        because they share a generic filename. Instead, the caller's real
        filename is additionally recorded, once, as `original_filename` on
        the document record `list_owned` (and anywhere else showing a
        private upload's name back to its owner) prefers over the internal
        storage-key-derived one. `None` (the default, and every existing
        caller before this fix -- KB ingestion never has a "caller's own
        filename" concept at all) falls back to `document.filename`, exactly
        today's behavior.

        `jurisdiction_metadata` is the flat dict produced by
        `kb_jurisdiction.document_metadata_fields` -- already validated by the
        caller (the admin ingestion service or the backfill script), never
        raw admin/user input. `None` (the default, and every existing caller
        before this phase) leaves every jurisdiction field absent, exactly as
        before -- an owner-scoped per-user upload never carries one.

        Failure-safe reindexing (P0-2): a source being RE-indexed (`previous`
        below is not `None`) never has its previously-indexed content taken
        offline before the new content is proven complete. The new version's
        chunks are written and validated with `metadata.document_status=
        "staging"` -- excluded from every retrieval path (`MongoVectorStore`/
        `BM25Index` both require `document_status="active"`, see those
        modules) -- and only flipped to `"active"` (with the old version
        flipped to `"superseded"` in the same step) once that validation
        passes. Any exception between claiming the per-source lock and that
        activation triggers cleanup of everything staged so far and re-raises;
        the previous version's chunks are never touched by a failed attempt.
        A crash that skips the cleanup (not a raised exception -- a killed
        process) leaves an inert `document_status="staging"` row that never
        becomes visible to retrieval; `DocumentVersionRepository.
        find_stale_staging` surfaces it for an explicit admin retry/cleanup,
        the same recovery shape `KnowledgeBaseStagingRepository.
        find_stale_active` already uses for the KB upload ledger.
        """
        document = await self.loader.load(path)
        self._clean_document_text(document)
        language = self.language_detector.detect(document.text[:4000])
        document.metadata["language"] = language
        document.metadata = await self.metadata_extractor.extract(document.text[:5000], document.metadata)
        if jurisdiction_metadata:
            document.metadata.update(jurisdiction_metadata)
        # Only a currently-serving version counts as "previous" -- an
        # abandoned `staging` row from a crashed attempt must not be picked up
        # as the version to increment from or supersede (see `latest_for_source`).
        owner_scope = {"owner_user_id": owner_user_id, "owner_session_id": owner_session_id}
        previous = await self.versions.latest_for_source(document.filename, **owner_scope) if (
            owner_user_id or owner_session_id
        ) else await self.versions.latest_for_source(document.filename)
        # A private attachment must remain uploadable in another conversation,
        # or after a failed review. Never reject it because another user (or
        # the shared KB) has the same bytes; that also leaks document presence.
        existing_same_hashes = set() if (owner_user_id or owner_session_id) else await self._known_hashes()
        # Re-indexing this same source with unchanged content (e.g. after a
        # chunker/pipeline code change) must not collide with its own current
        # version's hash -- duplicate detection is for a DIFFERENT source
        # carrying identical content, not this source updating itself.
        existing_same_hashes.discard((previous or {}).get("document_hash"))
        quality = await self.quality_checker.assess(path, document.text, document.metadata, existing_same_hashes)
        if not quality.passed:
            raise BadRequestError("Document failed indexing quality checks.", {"issues": quality.issues})
        version_number = int(previous.get("version_number", 0)) + 1 if previous else 1
        document.metadata.update(
            {
                "document_hash": quality.document_hash,
                "version_number": version_number,
                "index_status": "indexing",
                # Not yet "active": see the staging/activate sequence below.
                "document_status": "staging",
                "source_modified_time": path.stat().st_mtime,
                "embedding_model": settings.embedding_model,
                "embedding_version": settings.embedding_version,
                "vector_schema_version": settings.vector_schema_version,
                "namespace": document.metadata.get("legal_category", "general").lower().replace(" ", "_"),
                # Part 45 "Per-User Document Isolation": `None` for curated KB
                # documents (this is the only caller path that ever passes a
                # real value -- `IncrementalReindexRunner`'s calls, used only
                # for `storage/knowledge_base/`, never pass this argument) --
                # retrieval treats a `None`/absent owner as globally visible,
                # so curated documents and any document indexed before this
                # field existed keep working unchanged, no reindex required.
                "owner_session_id": owner_session_id,
                # Part 46 "Authenticated User Ownership": `None` unless the
                # uploader was authenticated (JWT-verified) at upload time --
                # when set, this is the authoritative ownership boundary
                # (survives across sessions, unlike `owner_session_id`).
                "owner_user_id": owner_user_id,
            }
        )

        # Cross-process "one in-flight reindex per source" lock -- mirrors
        # `KnowledgeBaseStagingRepository.claim_active`'s sparse-unique-index
        # pattern exactly. Claimed BEFORE the expensive chunk/embed work, and
        # before anything else durable is written, so a concurrent attempt for
        # the SAME source fails fast and cheaply instead of racing to activate.
        new_version_id = await self.versions.claim_staging(
            {
                "document_id": document.document_id,
                "source_document": document.filename,
                **owner_scope,
                "document_hash": quality.document_hash,
                "version_number": version_number,
                "effective_date": document.metadata.get("effective_date"),
                "published_date": document.metadata.get("published_date"),
                "updated_date": datetime.now(UTC).isoformat(),
                "government_source": document.metadata.get("government_source"),
                "index_status": "indexing",
                "document_status": "staging",
                "old_version_id": previous.get("_id") if previous else None,
                "new_version_id": None,
                "source_modified_time": path.stat().st_mtime,
                "quality_report": quality.__dict__,
                "runtime_versions": self.runtime_versions.snapshot(),
            }
        )
        if new_version_id is None:
            raise BadRequestError(
                f"Another indexing run is already in progress for {document.filename!r}."
            )

        documents_record_inserted = False
        activation_attempted = False
        previous_metadata_attempted = False
        old_document_id = (previous or {}).get("document_id")
        try:
            await self.documents.insert(
                {
                    "_id": document.document_id,
                    "filename": document.filename,
                    # Security/correctness finding K2: the caller's real
                    # filename, kept separate from `filename` above (the
                    # storage-key-derived identity `latest_for_source`/dedup
                    # key on) -- `None` when the caller has no such concept
                    # (KB ingestion), never silently defaulted to the
                    # storage key here so a reader can always tell "this
                    # document never had a distinct original name" apart
                    # from "the caller just didn't supply one this time".
                    "original_filename": original_filename,
                    "metadata": document.metadata,
                    "detected_language": language,
                    "chunk_count": 0,
                    "document_hash": quality.document_hash,
                    "version_number": version_number,
                    "index_status": "indexing",
                    "runtime_versions": self.runtime_versions.snapshot(),
                    "owner_session_id": owner_session_id,
                    "owner_user_id": owner_user_id,
                }
            )
            documents_record_inserted = True

            chunks = await self.chunker.chunk(document)
            if not chunks:
                raise BadRequestError("Chunking produced no content to index.")
            embedding_started = time.perf_counter()
            embeddings = await self.embeddings.embed_batch([chunk.text for chunk in chunks])
            log.info("indexing_embedding_timing", chunk_count=len(chunks),
                     embedding_ms=round((time.perf_counter() - embedding_started) * 1000, 2))
            enriched_chunks: list[DocumentChunk] = []
            current_chapter: str | None = None
            parent_section_metadata: dict[str, dict[str, Any]] = {}
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding
                # section_number/chapter are per-chunk facts, not document-wide ones (unlike
                # act_name). chunk.metadata was seeded from document.metadata, which already
                # carries whichever section/chapter happened to appear first in the document's
                # first 5000 chars -- left in place, extract()'s setdefault would silently keep
                # that stale value on every subsequent chunk instead of the chunk's own section.
                chunk.metadata.pop("section_number", None)
                chunk.metadata.pop("article_number", None)
                chunk.metadata.pop("chapter", None)
                # `section_number_provenance` must be popped alongside
                # `section_number` above, not left behind. It wasn't: a chunk
                # inheriting the document-level template's stale "heading"
                # provenance (set once from the document's first 5000 chars)
                # but finding no section heading of its OWN -- section_result
                # is None, so setdefault('section_number_provenance', ...)
                # below never even runs -- kept that stale "heading" value
                # while section_number itself correctly stayed absent.
                # Downstream, the carry-forward loop below reads exactly this
                # field to decide "did THIS chunk supply a real heading?"; the
                # stale value made it answer yes for a chunk that supplied
                # none, so it overwrote parent_section_metadata[parent_id]
                # with this chunk's own (empty) section_number instead of
                # taking the elif branch that copies the real one forward.
                # Confirmed live: BNS Section 303's punishment clause (page
                # 78, after 16 lettered illustrations) ended up with
                # section_number=None despite section 303's own heading chunk
                # sharing its parent_section_id and having the real value.
                chunk.metadata.pop("section_number_provenance", None)
                chunk.metadata = await self.metadata_extractor.extract(chunk.text, chunk.metadata)
                # Security/correctness finding C8: `language` has EXACTLY the
                # same "seeded from document.metadata, never re-derived per
                # chunk" shape the section_number/chapter/article_number
                # fields above were already fixed for -- `document.metadata
                # ["language"]` is detected once, from the document's first
                # 4000 characters (`index_file` above), and every chunk
                # silently inherited that single guess regardless of its own
                # actual text. A bilingual document (a Hindi Gazette
                # notification with an English cover page, an English Act
                # with an appended Hindi certified translation, ...) then had
                # EVERY chunk tagged with whichever language happened to open
                # the document -- including English chunks tagged Hindi, or
                # the reverse. Re-detected per chunk, from that chunk's own
                # text, the same way the document-level guess itself is
                # produced (`self.language_detector.detect`) -- never a
                # setdefault, since the stale value is always already
                # present and would otherwise never be overridden.
                chunk.metadata["language"] = self.language_detector.detect(chunk.text)
                parent_id = chunk.metadata.get("parent_section_id")
                if parent_id:
                    if chunk.metadata.get("section_number_provenance") == "heading":
                        parent_section_metadata[str(parent_id)] = {
                            key: chunk.metadata[key]
                            for key in ("section_number", "section_number_provenance", "article_number")
                            if key in chunk.metadata
                        }
                    elif str(parent_id) in parent_section_metadata:
                        chunk.metadata.update(parent_section_metadata[str(parent_id)])
                if jurisdiction_metadata:
                    # After extract() so section-level overrides (keyed by the
                    # section_number extract() just found on THIS chunk) can be
                    # applied -- see `kb_jurisdiction.apply_to_chunk`.
                    chunk.metadata = apply_to_chunk(chunk.metadata, jurisdiction_metadata)
                if chunk.metadata.get("chapter"):
                    current_chapter = chunk.metadata["chapter"]
                elif current_chapter:
                    # Chapters span many chunks with no heading of their own; carry the last
                    # seen one forward instead of leaving it unset for every chunk but the first.
                    chunk.metadata["chapter"] = current_chapter
                enriched_chunks.append(chunk)

            # Written under `document_id`, with document_status="staging" --
            # invisible to every retrieval path regardless of how long
            # activation takes or whether it ever runs. The previous version's
            # chunks are untouched by this call.
            await self.vector_store.upsert_chunks(enriched_chunks)

            # Validate the staged write is actually complete before switching
            # anything live -- a short count, not a full re-read, but enough to
            # catch a partial `bulk_write` (e.g. the process was killed mid-batch)
            # rather than activating a version that is missing chunks.
            staged_count = await self.vector_store.count_by_document_id(document.document_id)
            if staged_count != len(enriched_chunks):
                raise BadRequestError(
                    f"Staged write incomplete for {document.filename!r}: "
                    f"expected {len(enriched_chunks)} chunks, found {staged_count}."
                )

            # Activate: the new version's chunks become retrievable FIRST, the
            # old version's chunks are marked superseded SECOND -- so a crash
            # between the two steps leaves both briefly visible (harmless,
            # self-corrects on the next successful run or reconciliation pass)
            # rather than a window with neither visible.
            activation_attempted = True
            await self.vector_store.activate_version(document.document_id, old_document_id)
            if previous is not None:
                previous_metadata_attempted = True
                await self.versions.update_by_id(
                    str(previous["_id"]), {"new_version_id": new_version_id, "document_status": "superseded"}
                )
            await self.documents.update_by_id(
                document.document_id, {"index_status": "indexed", "chunk_count": len(enriched_chunks)}
            )
            # Release the source lock last, after all publication bookkeeping.
            await self.versions.activate(
                new_version_id, {"document_status": "active", "index_status": "indexed"}
            )
        except Exception:
            # Activation can fail after modifying old chunks, including during
            # BM25 persistence. Restore old content BEFORE deleting the new version.
            if activation_attempted and old_document_id:
                try:
                    await self.vector_store.activate_version(old_document_id, document.document_id)
                    if previous_metadata_attempted and previous:
                        await self.versions.update_by_id(
                            str(previous["_id"]), {"document_status": "active", "new_version_id": None}
                        )
                except Exception as rollback_exc:
                    log.error(
                        "reindex_rollback_failed", document_id=document.document_id,
                        old_document_id=old_document_id, error_type=type(rollback_exc).__name__,
                    )
                    # Do not delete potentially serving content without confirming
                    # restoration. Retain records for operator recovery.
                    raise
            # Anything staged so far is inert (document_status="staging" is
            # never retrievable) but is cleaned up anyway rather than left for
            # a human to notice -- a raised exception (unlike a killed
            # process) is always a chance to leave no trace of the failed
            # attempt. The previous, still-active version is never part of
            # this cleanup.
            try:
                await self.vector_store.delete_version_chunks(document.document_id)
            except Exception as cleanup_exc:  # noqa: BLE001 - best-effort; the original failure is what must propagate
                log.warning("reindex_cleanup_chunk_delete_failed", document_id=document.document_id, error=str(cleanup_exc))
            try:
                await self.versions.delete_by_id(new_version_id)
            except Exception as cleanup_exc:  # noqa: BLE001
                log.warning("reindex_cleanup_version_delete_failed", document_id=document.document_id, error=str(cleanup_exc))
            if documents_record_inserted:
                try:
                    await self.documents.collection.delete_one({"_id": document.document_id})
                except Exception as cleanup_exc:  # noqa: BLE001
                    log.warning("reindex_cleanup_document_delete_failed", document_id=document.document_id, error=str(cleanup_exc))
            raise

        # Best-effort physical cleanup of the now-superseded version's chunks.
        # Not required for correctness -- `document_status="superseded"` chunks
        # are already excluded from every retrieval path -- so a failure here
        # is logged, not raised: the new version is fully active regardless,
        # and `IndexReconciler`/a later reindex can clean up any leftovers.
        if previous is not None and old_document_id:
            try:
                await self.vector_store.delete_version_chunks(old_document_id)
            except Exception as cleanup_exc:  # noqa: BLE001
                log.warning(
                    "reindex_superseded_chunk_cleanup_failed",
                    source_document=document.filename,
                    old_document_id=old_document_id,
                    error=str(cleanup_exc),
                )

        return document.document_id, language, enriched_chunks

    async def deindex_source(self, source_document: str) -> bool:
        """Soft-deletes a source document that was removed from the knowledge base root.

        Marks the latest version and the document record as deleted and drops the
        associated vectors so they stop being retrieved, without deleting version
        history (audit trail is preserved).
        """
        latest = await self.versions.latest_for_source(source_document)
        if latest is None:
            return False
        await self.versions.update_by_id(str(latest["_id"]), {"document_status": "deleted", "index_status": "deindexed"})
        await self.vector_store.delete_by_source(source_document)
        document = await self.documents.collection.find_one({"filename": source_document})
        if document is not None:
            await self.documents.update_by_id(str(document["_id"]), {"index_status": "deleted"})
        return True

    async def _known_hashes(self) -> set[str]:
        cursor = self.versions.collection.find(
    {"document_status": {"$nin": ["deleted", "superseded"]},
     "owner_user_id": None, "owner_session_id": None},
    {"document_hash": 1},
)
        return {item["document_hash"] async for item in cursor if item.get("document_hash")}

    def _clean_text(self, text: str) -> str:
        lines = [line.strip() for line in text.replace("\x00", " ").splitlines()]
        # A wrapped sentence can start with "page boundary" or "confidential
        # information". Only discard recognizable standalone running headers.
        lines = [line for line in lines if line and line.casefold() != "confidential"
                 and not re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", line, re.IGNORECASE)]
        return "\n".join(lines)

    def _clean_document_text(self, document: LoadedDocument) -> None:
        """Cleans `document.text` while keeping `document.pages` in step with
        it (P0-1).

        Cleaning `document.text` after the loader built it from `document.
        pages` used to leave the per-page texts raw/uncleaned -- `_PageMap`
        (`app.rag.chunker`) only ever attributes a page when rejoining
        `document.pages` reproduces `document.text` EXACTLY, and a cleaned
        whole next to raw parts can practically never satisfy that, so every
        PDF chunk silently lost its page number the moment `_clean_text`
        removed even one blank line.

        The fix cleans each page's own text first and rebuilds `document.text`
        by rejoining the (now cleaned) pages with the loader's own separator --
        the two can never disagree, because `document.text` here IS that
        rejoin, not a separately-cleaned copy of it. `_clean_text` only
        strips/filters whole lines, so for the embedded-text case (separator
        "\\n", matching how the loader itself joined the pages) this produces
        the SAME final text as the old single-shot cleaning of the loader's own
        joined string -- page identity is added, no chunk boundary moves. The
        OCR case (separator "\\n\\n") keeps one literal blank line between
        pages that a single-shot clean used to swallow entirely along with
        every other blank line in the document; that boundary blank line is a
        harmless, even natural, paragraph break to the chunker's paragraph/
        section splitting, and is the unavoidable cost of a joined string that
        still proves which page each part came from.

        Non-paginated sources (TXT/DOCX/HTML/...) have no `pages` at all --
        cleaned exactly as before, no page identity invented for them.
        """
        if not document.pages:
            document.text = self._clean_text(document.text)
            return
        for page in document.pages:
            page.text = self._clean_text(page.text)
        separator = "\n\n" if document.metadata.get("ocr_applied") else "\n"
        document.text = separator.join(page.text for page in document.pages)
