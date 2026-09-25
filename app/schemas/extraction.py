"""Structured extraction contracts: entities, legal roles and timeline events.

Every field here exists to make an extraction CHECKABLE. A bare
`{"complainant": "Rahul Sharma"}` cannot be audited: it does not say what
text supported it, which page that text was on, whether a rule or a model
produced it, or how sure anything is. Those four facts are what separate an
extraction a lawyer can rely on from a guess that reads like one.

The rule the types enforce: a legal ROLE is never a property of a name, it is
a claim about a name that needs evidence. `ExtractedEntity.legal_role`
defaults to `"unknown"`, and a role only moves off that default when
`source_text` contains the words that assigned it. Everything the extractor
suspects but cannot support lands in `RoleCandidate`, separately, where it is
presented as a question rather than as a fact.
"""

from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal[
    "person",
    "organization",
    "court",
    "police_station",
    "address",
    "case_number",
    "fir_number",
    "date",
    "deadline",
    "amount",
    "obligation",
    "act",
    "section",
    "property",
    "vehicle_number",
    "document_reference",
    "email",
    "phone",
]

LegalRole = Literal[
    "unknown",
    "applicant",
    "respondent",
    "complainant",
    "accused",
    "petitioner",
    "defendant",
    "advocate",
    "witness",
    "landlord",
    "tenant",
    "employer",
    "employee",
    "buyer",
    "seller",
]

#: How a value was obtained. `rule` is a pattern that matched (evidence),
#: `conversation` is something the user already told us, `llm` is a model
#: proposal (never confirmed on its own), `fallback` is the deterministic
#: result used when a model call failed or returned unusable output.
ExtractionMethod = Literal["rule", "conversation", "llm", "fallback"]

EventCategory = Literal[
    "incident",
    "notice",
    "filing",
    "hearing",
    "payment",
    "agreement",
    "termination",
    "deadline",
    "other",
]

#: `unknown` is used whenever the event has no normalised date -- an undated
#: event has no expiry, and reporting one would be an invention.
ExpiryStatus = Literal["upcoming", "due_today", "expired", "unknown"]


class ExtractedEntity(BaseModel):
    """One value found in the text, with everything needed to check it."""

    value: str = Field(min_length=1)
    entity_type: EntityType
    legal_role: LegalRole = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    #: The span the value was read from. Required in practice for anything a
    #: user may act on: it is what lets them look and disagree.
    source_text: str = ""
    #: 1-based page in the source file, or `None` when the source carries no
    #: page identity. Never defaulted to 1.
    source_page: int | None = None
    extraction_method: ExtractionMethod = "rule"


class RoleCandidate(BaseModel):
    """A role the text HINTS at but does not establish.

    Kept apart from `ExtractedEntity` on purpose. "A vs B" strongly suggests
    a petitioner and a respondent, but which is which depends on the forum
    and the cause title, and a wrong guess puts the wrong person's name in
    the wrong half of a legal document.
    """

    value: str
    role: LegalRole
    evidence: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class RoleConflict(BaseModel):
    """The same name carrying two incompatible roles."""

    value: str
    roles: list[LegalRole]
    evidence: list[str] = Field(default_factory=list)

    @property
    def question(self) -> str:
        named = " and ".join(role.replace("_", " ") for role in self.roles)
        return (
            f'"{self.value}" appears as both {named} in this material. '
            "Which is correct? I will not choose for you."
        )


class TimelineEvent(BaseModel):
    """One dated (or explicitly undated) thing, with its provenance."""

    #: Exactly what the document said -- "on or before 15th March 2026",
    #: "within 30 days of receipt". Preserved verbatim because the phrasing
    #: is often what a deadline argument turns on.
    original_text: str = ""
    #: ISO `YYYY-MM-DD`, or `""` when no date could be established without
    #: guessing. Date-only by design: a hearing has a date, not an instant,
    #: and attaching a time would invent precision and a timezone.
    normalized_date: str = ""
    description: str = ""
    category: EventCategory = "other"
    #: Who must act, when the text says so. Empty otherwise -- never inferred.
    responsible_party: str = ""
    source_page: int | None = None
    is_deadline: bool = False
    expiry_status: ExpiryStatus = "unknown"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_text: str = ""
    #: For a relative deadline: what it runs FROM ("the date of receipt"),
    #: and how many days. Both empty/None for an absolute date.
    relative_to: str = ""
    relative_days: int | None = None
    extraction_method: ExtractionMethod = "rule"


class EntityExtractionResult(BaseModel):
    """Everything one extraction pass established, suspected, or could not settle."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    #: Roles suggested by the text but not established by it.
    unresolved_roles: list[RoleCandidate] = Field(default_factory=list)
    conflicts: list[RoleConflict] = Field(default_factory=list)
    #: Questions for the user, generated from conflicts and gaps. One at a
    #: time is the caller's job; this is the ordered backlog.
    questions: list[str] = Field(default_factory=list)
    #: How many values each method contributed, for auditing an extraction
    #: after the fact ("was that name a rule match or a model guess?").
    method_counts: dict[str, int] = Field(default_factory=dict)

    def of_type(self, entity_type: EntityType) -> list[ExtractedEntity]:
        return [entity for entity in self.entities if entity.entity_type == entity_type]

    def with_role(self, role: LegalRole) -> list[ExtractedEntity]:
        return [entity for entity in self.entities if entity.legal_role == role]


class TimelineExtractionResult(BaseModel):
    """A chronology, plus what could not be placed on it."""

    #: Dated events, earliest first.
    events: list[TimelineEvent] = Field(default_factory=list)
    #: Events with no establishable date. Kept separately rather than being
    #: given a placeholder date, which would put them in a false order.
    undated_events: list[TimelineEvent] = Field(default_factory=list)
    #: Dates that cannot all be true (an event after its own deadline, a
    #: filing before the incident). Reported, never silently resolved.
    contradictions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
