"""Regression test for correctness finding C8 (English documents tagged
Hindi, and the reverse).

Root cause: `IndexingPipeline.index_file` detects a document's language ONCE,
from only its first 4000 characters (`document.metadata["language"] =
self.language_detector.detect(document.text[:4000])`), and every chunk's
metadata is seeded from that single document-level dict. `section_number`/
`article_number`/`chapter` were already fixed for the identical shape of bug
(a per-chunk fact silently inheriting a document-level guess) -- `language`
was not, so a bilingual document (an English Act with an appended Hindi
certified translation, a Hindi Gazette notification with an English cover
page, ...) tagged EVERY chunk with whichever language happened to open the
document, regardless of that chunk's own actual text.

Follows `tests/test_kb_jurisdiction_pipeline.py`'s convention: the real
`DocumentLoader`/`SectionAwareChunker`/`LanguageDetector` run over a plain-
text fixture; only the Mongo-backed repositories/vector store are faked.
"""

import asyncio
from typing import Any
from uuid import uuid4

from app.rag.chunker import SectionAwareChunker
from app.rag.loader import DocumentLoader
from app.rag.pipeline import IndexingPipeline
from app.rag.quality import DocumentQualityChecker


class _EmptyCursor:
    def __aiter__(self) -> Any:
        async def _gen() -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

        return _gen()


class _FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list[Any] = []

    async def upsert_chunks(self, chunks: list[Any]) -> None:
        self.upserted.extend(chunks)

    async def delete_by_source(self, source_document: str) -> int:
        return 0

    async def count_by_document_id(self, document_id: str) -> int:
        return sum(1 for chunk in self.upserted if chunk.document_id == document_id)

    async def activate_version(self, new_document_id: str, old_document_id: str | None) -> None:
        for chunk in self.upserted:
            if chunk.document_id == new_document_id:
                chunk.metadata["document_status"] = "active"
            elif old_document_id and chunk.document_id == old_document_id:
                chunk.metadata["document_status"] = "superseded"

    async def delete_version_chunks(self, document_id: str) -> int:
        before = len(self.upserted)
        self.upserted = [chunk for chunk in self.upserted if chunk.document_id != document_id]
        return before - len(self.upserted)


class _FakeEmbeddings:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeCollection:
    def find(self, *_args: Any, **_kwargs: Any) -> Any:
        return _EmptyCursor()


class _FakeVersionRepository:
    def __init__(self) -> None:
        self.collection = _FakeCollection()
        self.inserted: list[dict[str, Any]] = []

    async def latest_for_source(self, _source_document: str, **_scope: Any) -> dict[str, Any] | None:
        return None

    async def insert(self, document: dict[str, Any]) -> str:
        self.inserted.append(document)
        return str(uuid4())

    async def update_by_id(self, *_args: Any, **_kwargs: Any) -> bool:
        return True

    async def claim_staging(self, document: dict[str, Any]) -> str | None:
        return await self.insert(document)

    async def activate(self, version_id: str, updates: dict[str, Any]) -> bool:
        return True

    async def delete_by_id(self, version_id: str) -> bool:
        return True


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.inserted: list[dict[str, Any]] = []
        self.collection = _FakeCollection()

    async def insert(self, document: dict[str, Any]) -> str:
        self.inserted.append(document)
        return str(document["_id"])

    async def update_by_id(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


def _build_pipeline() -> tuple[IndexingPipeline, _FakeVectorStore]:
    vector_store = _FakeVectorStore()
    pipeline = IndexingPipeline(
        loader=DocumentLoader(),
        chunker=SectionAwareChunker(),
        embeddings=_FakeEmbeddings(),
        vector_store=vector_store,
    )
    pipeline.documents = _FakeDocumentRepository()  # type: ignore[assignment]
    pipeline.versions = _FakeVersionRepository()  # type: ignore[assignment]
    pipeline.quality_checker = DocumentQualityChecker()
    return pipeline, vector_store


# A long, entirely genuine-looking English preamble -- well past the 4000
# characters `index_file` samples for its ONE document-level language guess
# -- followed by a section that is genuinely, unambiguously Hindi.
_ENGLISH_PREAMBLE_SECTIONS = "\n\n".join(
    f"{number}. Definitions and general provisions.—In this Act, unless the context otherwise requires, "
    "the expression used in this section shall have the meaning assigned to it by the rules made under "
    "this Act, and every reference to a designated authority shall be construed as a reference to the "
    "officer notified for the purposes of this Chapter by the appropriate Government from time to time."
    for number in range(1, 40)
)
_HINDI_SECTION = (
    "999. हिंदी अनुभाग।—इस अधिनियम की धारा 12 के अंतर्गत दायर की गई किसी भी शिकायत का निपटारा "
    "निर्धारित प्राधिकारी द्वारा तीस दिनों के भीतर किया जाएगा, और इस संबंध में कोई भी विलंब "
    "उचित कारण के बिना स्वीकार्य नहीं होगा।"
)
_BILINGUAL_ACT_TEXT = f"{_ENGLISH_PREAMBLE_SECTIONS}\n\n{_HINDI_SECTION}\n"


def test_a_later_hindi_section_is_not_tagged_english_from_the_documents_opening(tmp_path) -> None:
    assert len(_ENGLISH_PREAMBLE_SECTIONS) > 4000, "the preamble must dominate the document-level 4000-char sample"
    source = tmp_path / "bilingual_act.txt"
    source.write_text(_BILINGUAL_ACT_TEXT, encoding="utf-8")
    pipeline, _vector_store = _build_pipeline()

    document_id, document_language, chunks = asyncio.run(pipeline.index_file(source))

    assert document_id
    # The document-level guess (from the first 4000 chars) is legitimately
    # English -- that alone is not the bug.
    assert document_language == "english"
    hindi_chunks = [chunk for chunk in chunks if "धारा 12" in chunk.text]
    assert hindi_chunks, "the Hindi section must have produced its own chunk"
    for chunk in hindi_chunks:
        assert chunk.metadata["language"] == "hindi", (
            f"chunk {chunk.chunk_id!r} contains Hindi text but is tagged {chunk.metadata['language']!r}"
        )
    english_chunks = [chunk for chunk in chunks if chunk not in hindi_chunks]
    assert all(chunk.metadata["language"] == "english" for chunk in english_chunks)
