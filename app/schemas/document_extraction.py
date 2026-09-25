from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class ExtractionBlock(BaseModel):
    kind: Literal["heading", "paragraph", "list_item", "table"]
    text: str = Field(default="", max_length=30000)
    level: int = Field(default=1, ge=1, le=6)
    rows: list[list[str]] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        if any(len(row) > 30 or any(len(cell) > 5000 for cell in row) for row in self.rows):
            raise ValueError("Table is too large.")
        if self.kind == "table" and (not self.rows or not any(self.rows)):
            raise ValueError("Table rows are missing.")
        return self


class ExtractedPage(BaseModel):
    page_number: int | None = Field(default=None, ge=1)
    blocks: list[ExtractionBlock] = Field(default_factory=list, max_length=500)
    warnings: list[str] = Field(default_factory=list, max_length=50)


class DocumentExtractionResponse(BaseModel):
    filename: str
    pages: list[ExtractedPage]
    text: str
    docx_base64: str
    warnings: list[str]
    extraction_method: str = "vision_ocr"
