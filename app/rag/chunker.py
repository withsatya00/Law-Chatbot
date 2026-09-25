import hashlib
import re
from uuid import uuid4

from app.rag.types import DocumentChunk, LoadedDocument


class _PageMap:
    """Maps a character offset in the joined document text back to its page.

    The chunker splits `document.text`, which the loader built by joining the
    per-page texts. Rather than re-splitting text (which would fight the
    section-aware boundaries this chunker exists to respect), each page's
    character span in that joined string is recorded once, and a chunk's page
    is then the span its text actually lands in.

    Built from the SAME separator the loader used, so the offsets are exact
    rather than estimated. If a document is not paginated the map is empty and
    every lookup returns `None` -- a TXT file has no page 1, and saying it does
    would be an invented citation.
    """

    def __init__(self, document: LoadedDocument) -> None:
        self._spans: list[tuple[int, int, int, str]] = []
        if not document.pages:
            return
        # The loader joins embedded-text pages with "\n" and OCR pages with
        # "\n\n"; derive which from the document itself rather than guessing.
        separator = "\n\n" if document.metadata.get("ocr_applied") else "\n"
        # The offsets are only meaningful if the pages actually reconstruct the
        # text being chunked. A `LoadedDocument` assembled elsewhere -- or one
        # whose text was post-processed after loading -- would otherwise yield
        # spans pointing at the wrong places, and every page number derived from
        # them would be confidently wrong. Verify, and decline to attribute any
        # page at all when it does not hold.
        if separator.join(page.text for page in document.pages) != document.text:
            return
        cursor = 0
        for page in document.pages:
            start = cursor
            end = start + len(page.text)
            self._spans.append((start, end, page.page_number, page.extraction_method))
            cursor = end + len(separator)

    def locate(self, text: str, search_from: int, haystack: str) -> tuple[int | None, int | None, str | None, int]:
        """`(first_page, last_page, extraction_method, next_search_offset)`.

        `search_from` makes the scan forward-only: chunks are emitted in
        document order, so a later chunk can never match an earlier page's copy
        of a repeated heading.
        """
        if not self._spans:
            return None, None, None, search_from
        probe = text[:200].strip()
        offset = haystack.find(probe, search_from) if probe else -1
        if offset < 0:
            offset = haystack.find(probe) if probe else -1
        if offset < 0:
            # The chunk text was transformed (overlap carry, large-split) and no
            # longer appears verbatim. No page can be attributed with
            # confidence, so none is claimed.
            return None, None, None, search_from
        end = offset + len(text)
        touched = [span for span in self._spans if span[0] < end and span[1] > offset]
        if not touched:
            return None, None, None, offset
        methods = {span[3] for span in touched if span[3] != "none"}
        method = methods.pop() if len(methods) == 1 else ("mixed" if methods else None)
        return touched[0][2], touched[-1][2], method, offset


class SectionAwareChunker:
    def __init__(self, max_chars: int = 2400, overlap_chars: int = 250) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    async def chunk(self, document: LoadedDocument) -> list[DocumentChunk]:
        self._page_map = _PageMap(document)
        self._page_map_document_id = document.document_id
        self._search_offset = 0
        sections, has_clean_headings = self._split_on_legal_boundaries(document.text)
        if has_clean_headings:
            # Every section already has its own real heading boundary -- keep each
            # one as its own chunk instead of packing several into one buffer.
            # Packing was silently discarding section_number metadata for every
            # section but the first in a shared buffer (confirmed: 131/303 BNS
            # sections, 43%, lost their tag this way -- MetadataExtractor._section_number()
            # only ever records the FIRST heading match per chunk, so a merged
            # buffer can represent at most one section).
            clean_chunks: list[DocumentChunk] = []
            for section in sections:
                if len(section) <= self.max_chars:
                    clean_chunks.append(self._make_chunk(document, section))
                else:
                    parts = self._split_large(section)
                    parent_id = self._parent_section_id(document, section)
                    heading = section.splitlines()[0].strip()[:300]
                    for index, part in enumerate(parts):
                        clean_chunks.append(self._make_chunk(document, part, {
                            "parent_section_id": parent_id,
                            "section_part_index": index + 1,
                            "section_part_count": len(parts),
                            "section_heading": heading,
                        }))
            return clean_chunks
        chunks: list[DocumentChunk] = []
        buffer = ""
        for section in sections:
            candidate = f"{buffer}\n\n{section}".strip()
            if len(candidate) <= self.max_chars:
                buffer = candidate
                continue
            if buffer:
                chunks.append(self._make_chunk(document, buffer))
                # The tail-overlap + this section might ITSELF still exceed max_chars
                # (e.g. a table with no blank-line breaks, like BNSS's First Schedule
                # offence-classification table, comes back as one multi-hundred-KB
                # "section" from _split_on_legal_boundaries's paragraph-split fallback).
                # Route it through _split_large instead of assuming a single flush at
                # the end of the loop will catch an oversized buffer -- previously it
                # didn't: an oversized buffer left dangling here only got caught by the
                # unconditional `if buffer: chunks.append(...)` after the loop, which
                # emitted it whole with no size check at all.
                carried = f"{buffer[-self.overlap_chars:]}\n\n{section}".strip()
                if len(carried) <= self.max_chars:
                    buffer = carried
                else:
                    for part in self._split_large(carried):
                        chunks.append(self._make_chunk(document, part))
                    buffer = ""
            else:
                for part in self._split_large(section):
                    chunks.append(self._make_chunk(document, part))
                buffer = ""
        if buffer:
            chunks.append(self._make_chunk(document, buffer))
        return chunks

    def _split_on_legal_boundaries(self, text: str) -> tuple[list[str], bool]:
        # Prefixed headings ("Section 173", "Article 21") cover most sources, but acts
        # like BNS/BNSS/BSA number their own sections bare ("318. Cheating.—...", no
        # "Section" prefix) -- the em/en-dash requirement (mirroring SECTION_HEADING_RE)
        # keeps this from also splitting on TOC entries and footnote/cross-reference
        # numbers, which don't have the dash-separated operative body.
        # A third shape (mirroring SECTION_HEADING_NO_TITLE_RE in metadata.py):
        # some official sources open a section straight on subsection (1) with no
        # title/dash at all ("173. (1) Every information..." -- confirmed live in
        # BNSS_2023_Official_Gazette.pdf). Without this branch the whole document
        # never split on a section boundary at all, so unrelated sections were
        # packed into one buffer and every section after the first silently lost
        # its own identity.
        pattern = (
            r"(?=\n\s*(?:Section|Sec\.|Article|Rule|Chapter)\s+[0-9A-ZIVXLC])"
            r"|(?=\n[ \t]*\d{1,4}[A-Z]{0,2}\.\s+[A-Z][^\n]{0,140}?[—–])"
            r"|(?=\n[ \t]*\d{1,4}[A-Z]{0,2}\.\s*\(1\))"
        )
        parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
        starts_with_heading = bool(re.match(
            r"\s*(?:(?:Section|Sec\.|Article|Rule|Chapter)\s+[0-9A-ZIVXLC]|\d{1,4}[A-Z]{0,2}\.\s+[A-Z]|\d{1,4}[A-Z]{0,2}\.\s*\(1\))",
            text,
        ))
        if len(parts) <= 1 and not starts_with_heading:
            # No clean numbered-heading boundaries found -- fall back to paragraph
            # splitting and let chunk() pack multiple paragraphs per chunk as before
            # (unchanged behavior for FAQs, judgments, commentary, etc.).
            return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()], False
        return parts, True

    def _split_large(self, text: str) -> list[str]:
        """Split at paragraph/sentence/word boundaries where possible.

        The overlap remains verbatim so page attribution can still locate each
        part in the original document. Hard character cuts are a final fallback
        for tables or OCR runs with no usable boundary.
        """
        if len(text) <= self.max_chars:
            return [text]
        parts: list[str] = []
        start = 0
        minimum = max(self.max_chars // 2, self.overlap_chars + 1)
        while start < len(text):
            hard_end = min(start + self.max_chars, len(text))
            end = hard_end
            if hard_end < len(text):
                window = text[start + minimum:hard_end]
                candidates = [window.rfind("\n\n"), window.rfind(". "), window.rfind("; "), window.rfind(" ")]
                boundary = next((candidate for candidate in candidates if candidate >= 0), -1)
                if boundary >= 0:
                    end = start + minimum + boundary + (2 if window[boundary:boundary + 2] in {". ", "; "} else 0)
            part = text[start:end].strip()
            if part:
                parts.append(part)
            if end >= len(text):
                break
            start = max(end - self.overlap_chars, start + 1)
        return parts

    @staticmethod
    def _parent_section_id(document: LoadedDocument, section: str) -> str:
        digest = hashlib.sha256(f"{document.document_id}\0{section[:500]}".encode()).hexdigest()[:24]
        return f"section:{digest}"

    def _make_chunk(
        self, document: LoadedDocument, text: str, extra_metadata: dict[str, object] | None = None,
    ) -> DocumentChunk:
        metadata = dict(document.metadata)
        if extra_metadata:
            metadata.update(extra_metadata)
        page_map = self._page_map_for(document)
        first, last, method, offset = page_map.locate(text, self._search_offset, document.text)
        self._search_offset = offset
        # Only ever set when a page was actually determined. Absent keys read as
        # "this source has no page evidence", which is the honest answer for a
        # TXT/DOCX/HTML source and for a chunk whose text could not be located
        # verbatim in the joined document.
        if first is not None:
            metadata["page_number"] = first
            metadata["page_start"] = first
            metadata["page_end"] = last
            if last is not None and last != first:
                metadata["page_range"] = f"{first}-{last}"
            if method:
                metadata["extraction_method"] = method
        return DocumentChunk(
            chunk_id=str(uuid4()),
            document_id=document.document_id,
            text=text,
            metadata=metadata,
        )

    def _page_map_for(self, document: LoadedDocument) -> _PageMap:
        # Rebuilt whenever a different document arrives. `chunk()` is the only
        # public entry point and processes one document at a time, so a cache
        # keyed on document_id is enough and keeps the map off the constructor
        # (this chunker is shared as a long-lived instance by the pipeline).
        if getattr(self, "_page_map_document_id", None) != document.document_id:
            self._page_map = _PageMap(document)
            self._page_map_document_id = document.document_id
            self._search_offset = 0
        return self._page_map
