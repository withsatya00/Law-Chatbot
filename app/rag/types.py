from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


ExtractionMethod = Literal["embedded_text", "ocr", "none"]


class LoadedPage(BaseModel):
    """One page of a paginated source, with its ORIGINAL page number.

    `page_number` is 1-based and is the number the page has in the file, not
    its position in this list. The two diverge the moment a page yields no
    text: dropping such a page and re-numbering would silently shift every
    citation after it, so an empty page is kept with `extraction_method="none"`
    instead of being removed.
    """

    page_number: int = Field(ge=1)
    text: str
    extraction_method: ExtractionMethod = "embedded_text"


class LoadedDocument(BaseModel):
    document_id: str
    filename: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Populated only for paginated formats (PDF today). Empty for TXT, DOCX,
    # HTML and the rest, whose chunks legitimately carry `page_number=None` --
    # those formats have no page identity to preserve, and inventing one would
    # be worse than admitting there is none.
    pages: list[LoadedPage] = Field(default_factory=list)

    @property
    def is_paginated(self) -> bool:
        return bool(self.pages)
