from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class EvidencePage(BaseModel):
    """One page a cited source was actually read from.

    Surfaced separately from `SourceCitation` so a client can render "check
    page 42 of bns.pdf" without re-deriving it from every citation, and so the
    absence of page evidence is explicit: an empty `evidence_pages` list means
    the answer rests on sources with no page identity (a plain-text source, or
    a chunk indexed before page capture existed), NOT that the answer is
    unsourced. `SourceCitation` remains the authority on what was cited.
    """

    source_document: str
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    extraction_method: str | None = None
    label: str = ""


class SourceCitation(BaseModel):
    """One retrieved source, as shown to the user.

    Phase 1 item 2: `label` is a single human-readable rendering of whichever
    identifying fields this source actually has -- "Bharatiya Nyaya Sanhita,
    2023 — Section 318 (bns_2023.pdf)". Every consumer (the chat API, the
    Streamlit UI, the answer-quality gate) previously had to reassemble that
    from the raw fields, and mostly didn't: the UI dumped the JSON and the LLM
    fell back to citing "Source 2", an internal prompt index that means
    nothing to a reader. Computed in a validator rather than by the callers so
    it is populated no matter who constructs the model.
    """

    act_name: str | None = None
    section: str | None = None
    article: str | None = None
    chapter: str | None = None
    source_document: str
    government_source: str | None = None
    url: str | None = None
    label: str = ""
    source_version: str | None = None
    effective_date: str | None = None
    amendment_status: str | None = None
    last_verified_date: str | None = None
    verification_status: str | None = None
    current_as_of: str | None = None
    # Page-level evidence. All optional and all default to None, because most
    # of the existing corpus predates page capture and non-paginated formats
    # (TXT/DOCX/HTML) have no page identity at all. `None` here means "this
    # source carries no page evidence", never "page 1".
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    # "embedded_text" | "ocr" | "mixed" -- whether the cited page's text came
    # from the PDF's own text layer or from OCR of a scan. A reader checking a
    # citation deserves to know which, since OCR text can be imperfect.
    extraction_method: str | None = None

    @property
    def page_reference(self) -> str | None:
        """A human-readable page reference, or `None` when there is no page
        evidence. Deliberately returns `None` rather than an empty string so a
        caller cannot render "page " with nothing after it."""
        if self.page_start is None:
            return f"page {self.page_number}" if self.page_number is not None else None
        if self.page_end is not None and self.page_end != self.page_start:
            return f"pages {self.page_start}-{self.page_end}"
        return f"page {self.page_start}"

    @model_validator(mode="after")
    def _fill_label(self) -> "SourceCitation":
        if not self.label:
            self.label = build_citation_label(
                act_name=self.act_name,
                section=self.section,
                article=self.article,
                chapter=self.chapter,
                source_document=self.source_document,
                url=self.url,
            )
            page_reference = self.page_reference
            if page_reference:
                self.label += f" · {page_reference}"
            if self.source_version:
                self.label += f" · version {self.source_version}"
            if self.current_as_of:
                self.label += f" · current as of {self.current_as_of}"
            if self.amendment_status in {"repealed", "superseded"}:
                self.label += f" · {self.amendment_status.upper()}"
        return self

    @property
    def is_identifiable(self) -> bool:
        """Whether this citation names something a reader could actually look
        up. A chunk whose metadata carries only an internal filename with no
        Act, section or article is not a legal citation -- the answer-quality
        gate treats an answer backed by nothing else as ungrounded."""
        return bool(self.act_name or self.section or self.article or self.url or self.government_source)


def build_citation_label(
    *,
    act_name: str | None = None,
    section: str | None = None,
    article: str | None = None,
    chapter: str | None = None,
    source_document: str | None = None,
    url: str | None = None,
) -> str:
    """A readable citation from whichever identifying fields are present.

    Never invents an Act name: a source with only a filename renders as that
    filename, which is honest, rather than as a guessed statute.
    """
    provision_parts: list[str] = []
    if section:
        provision_parts.append(f"Section {section}")
    if article:
        provision_parts.append(f"Article {article}")
    if not provision_parts and chapter:
        provision_parts.append(str(chapter))

    head = act_name.strip() if act_name else ""
    provision = ", ".join(provision_parts)
    if head and provision:
        label = f"{head} — {provision}"
    else:
        label = head or provision

    if source_document:
        label = f"{label} ({source_document})" if label else str(source_document)
    if url:
        label = f"{label} · {url}" if label else str(url)
    return label or "Unattributed source"


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LawyerRecommendation(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    urgency: str = "routine"
    jurisdiction_note: str = "Choose an advocate entitled to practise in the relevant Indian court or forum."
    documents_to_carry: list[str] = Field(default_factory=list)
    questions_to_ask: list[str] = Field(default_factory=list)
    directory_available: bool = False
    directory_notice: str = (
        "No verified lawyer directory is connected. This is a specialization suggestion, not a lawyer listing."
    )
    verified_listings: list["LawyerDirectoryListing"] = Field(default_factory=list)


class LawyerDirectoryListing(BaseModel):
    """A person returned by a configured, verified directory provider.

    Application or model output must never be used to construct one of these
    records.  Every field is copied from the external directory response.
    """

    full_name: str
    enrolment_number: str = ""
    verified_by: str = ""
    verified_on: str = ""
    practice_areas: list[str] = Field(default_factory=list)
    city: str = ""
    state: str = ""
    languages: list[str] = Field(default_factory=list)
    contact: str = ""
    profile_url: str = ""


class HealthComponent(BaseModel):
    status: str
    latency_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TimestampedResponse(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)


def evidence_pages_from_citations(citations: "list[SourceCitation]") -> list[EvidencePage]:
    """The subset of `citations` that carry real page evidence.

    Citations without a page are omitted rather than emitted with `None`, so a
    caller can treat a non-empty list as "these are checkable page references"
    and an empty one as "no page evidence is available for this answer".
    """
    pages: list[EvidencePage] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for citation in citations:
        if citation.page_number is None and citation.page_start is None:
            continue
        key = (citation.source_document, citation.page_start, citation.page_end)
        if key in seen:
            continue
        seen.add(key)
        pages.append(
            EvidencePage(
                source_document=citation.source_document,
                page_number=citation.page_number,
                page_start=citation.page_start,
                page_end=citation.page_end,
                extraction_method=citation.extraction_method,
                label=citation.page_reference or "",
            )
        )
    return pages
