from pydantic import BaseModel, Field


class TranscribeParty(BaseModel):
    name: str = ""
    address: str = ""
    phone: str = ""
    email: str = ""


class TranscribeVerificationIssue(BaseModel):
    field: str = ""
    value: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class DocumentTranscribeResponse(BaseModel):
    """A source document (often handwritten/scanned) retyped into a clean,
    professionally formatted version, with its facts preserved verbatim --
    not a summary, and never invented content. See `document_transcribe_prompt.md`
    for the full no-fabrication contract this is validated against.
    """

    document_type: str = "Unknown"
    primary_language: str = ""
    languages_detected: list[str] = Field(default_factory=list)
    handwritten: bool = False
    applicant: TranscribeParty = Field(default_factory=TranscribeParty)
    respondent: TranscribeParty = Field(default_factory=TranscribeParty)
    subject: str = ""
    reference: str = ""
    facts: list[str] = Field(default_factory=list)
    grounds: list[str] = Field(default_factory=list)
    legal_provisions: list[str] = Field(default_factory=list)
    relief_requested: str = ""
    annexures: list[str] = Field(default_factory=list)
    place: str = ""
    date: str = ""
    signature_present: bool = False
    formatted_text: str = ""
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_required: bool = False
    issues: list[TranscribeVerificationIssue] = Field(default_factory=list)
