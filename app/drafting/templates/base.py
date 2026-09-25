from dataclasses import dataclass, field
from typing import Any, Literal

FieldType = Literal["text", "textarea", "date", "tel", "email", "number"]

# Part 42 "Draft Formatting": one generic 15-heading skeleton (Court/Authority
# + Title + ... + Annexure List) used to be forced onto every document type,
# which is why a Legal Notice — addressed to a private party, no court
# involved — still showed a "Court / Authority Name" heading, and every
# draft carried boilerplate like "Legal Grounds: As set out in the facts
# stated above, read with the applicable law." regardless of whether it
# added anything. Replaced with a small, fixed set of structural skeletons
# keyed by `DraftTemplateDefinition.category` (Notice/Complaint/Affidavit/
# Application — the categories every template YAML already declares), each
# shaped like the professional document a reader actually expects for that
# category. "Annexures" is deliberately excluded from every skeleton below:
# it is appended only when a template's `available_documents` field is
# actually non-empty (see `LegalDraftEngine`), never as an empty "None."
# placeholder heading.
#
# Part 56 "Advocate-Style Draft Redesign": Notice and Complaint now share one
# UNIFIED_SECTIONS skeleton (previously two separate, narrower skeletons) --
# an advocate-style 600-900 word document needs an explicit Introduction,
# Facts of the Case (multi-paragraph), Legal Position, and Consequences
# section ahead of the Prayer, none of which the old terse "Notice"/"Facts"+
# "Request" single-paragraph shape had room for. Affidavit and Application
# (RTI) deliberately KEEP their own distinct, narrower skeletons -- an
# affidavit is a sworn numbered statement with a verification/oath clause,
# not a grievance narrative with a "Prayer", and an RTI application is a
# numbered Section 6(1) information request, not a Legal Position/
# Consequences argument. Both still get the same word-count expansion and
# legal-enrichment treatment, just organized under their own existing
# headings (see `LegalDraftEngine._render_sections`'s category-conditional
# `narrative_guidance`).
#
# Part 53 "PDF Hindi Rendering + Professional Layout Audit": "Place" and
# "Date" are deliberately NOT separate headings for Notice/Complaint/
# Application -- a real Indian legal letter's closing block reads "स्थान:
# लखनऊ" / "दिनांक: 13/08/2026" as inline-labeled lines directly above the
# signature, not as their own headed sections partway through the document.
# Both are folded into "Signature Block" (built by
# `LegalDraftEngine._closing_block`, which unconditionally overrides whatever
# the LLM/deterministic path wrote for this section, the same way "Date" was
# already always forced before this change). Affidavit keeps its own
# separate "Place"/"Date" -- a sworn statement's closing convention is
# different and wasn't part of this fix.
NOTICE_SECTIONS: tuple[str, ...] = (
    "Recipient",
    "Subject",
    # A demand notice is sent BY one private party TO another. Its author is
    # the sender, not a "complainant" -- that word belongs to a document
    # addressed to a police station or a consumer forum. The Notice skeleton
    # reused "Complainant Details" purely because it was cloned from the
    # Complaint skeleton, and the effect was visible in real output: a Hindi
    # recovery notice headed the sender's own name and address
    # "परिवादी का विवरण" ("Complainant's details"), which is simply the wrong
    # legal term for that document.
    "Sender Details",
    "Introduction",
    "Facts of the Case",
    "Legal Position",
    "Consequences",
    "Prayer",
    "Signature Block",
)
COMPLAINT_SECTIONS: tuple[str, ...] = (
    "Recipient",
    "Subject",
    "Complainant Details",
    "Introduction",
    "Facts of the Case",
    "Legal Position",
    "Consequences",
    "Prayer",
    "Verification",
    "Signature Block",
)
AFFIDAVIT_SECTIONS: tuple[str, ...] = (
    "Court / Authority Name", "Deponent Details", "Statements", "Verification", "Place", "Date", "Signature",
)
APPLICATION_SECTIONS: tuple[str, ...] = (
    "To", "Subject", "Applicant Details", "Request", "Signature",
)
# Part 57 "Drafting Lifecycle Redesign": agreement/contract-style documents
# (NDA, MOU, Partnership/Service Agreement, Rent Agreement, Property Sale/
# Purchase Agreement) don't fit any skeleton above -- they're multi-party
# clause-based instruments (recitals, numbered terms, a term/termination
# clause, two signature blocks), not a single-applicant letter/petition to
# a recipient or authority. Kept as its own category rather than stretching
# UNIFIED_SECTIONS to cover both shapes.
CONTRACT_SECTIONS: tuple[str, ...] = (
    "Title", "Parties", "Recitals", "Terms and Conditions", "Term and Termination",
    "Governing Law and Jurisdiction", "Signatures",
)

CATEGORY_SECTIONS: dict[str, tuple[str, ...]] = {
    "Notice": NOTICE_SECTIONS,
    "Complaint": COMPLAINT_SECTIONS,
    "Affidavit": AFFIDAVIT_SECTIONS,
    "Application": APPLICATION_SECTIONS,
    "Contract": CONTRACT_SECTIONS,
}

# A section not in the category's own skeleton but allowed to appear anyway
# when it has real content (checked in `LegalDraftEngine`/`_parse_sections`).
# "Enclosures" is the Complaint/Application-category name for the same
# concept "Annexures" names elsewhere -- see `advocate_register.py`.
OPTIONAL_SECTIONS: tuple[str, ...] = ("Annexures", "Enclosures")


# Problem-first discovery (see `app/drafting/recommendation.py` and
# `app/drafting/discovery.py`). These are the closed vocabularies the YAML
# loader validates new, optional per-template metadata against -- kept here
# rather than duplicated in the loader/recommendation modules so there is one
# source of truth both import.
#
# `document_family` is a broader structural classification than `category`
# (which still drives which heading skeleton is rendered -- see
# `structure_sections_for`/`CATEGORY_SECTIONS` above and is NOT repurposed).
# `document_family` exists so future template types (a pleading, a deed, a
# will) can be tagged meaningfully for search/recommendation even before a
# dedicated structural skeleton exists for them; a template whose family has
# no skeleton yet still renders through its `category`'s skeleton unchanged.
DOCUMENT_FAMILIES: tuple[str, ...] = (
    "notice", "complaint", "affidavit", "application", "contract", "pleading",
    "court_application", "petition", "appeal", "reply", "deed",
    "corporate_resolution", "will", "policy", "regulatory_form",
)

# The case-stage vocabulary the discovery conversation asks about and
# `DraftTemplateDefinition.case_stages` is scored against.
CASE_STAGES: tuple[str, ...] = (
    "pre_litigation", "notice_sent", "reply_received", "case_pending",
    "evidence_stage", "order_passed", "appeal", "execution",
)

TEMPLATE_STATUSES: tuple[str, ...] = ("draft", "production", "deprecated")


def structure_sections_for(category: str) -> tuple[str, ...]:
    """The ordered heading skeleton for a template's category. Falls back to
    the unified Notice/Complaint skeleton for any category not in the fixed
    set above, rather than raising — a new template YAML with an
    unrecognized `category` should still degrade to a reasonable document
    instead of crashing.
    """
    return CATEGORY_SECTIONS.get(category, NOTICE_SECTIONS)


@dataclass(frozen=True)
class DraftField:
    key: str
    label: str
    hindi_label: str
    field_type: FieldType = "text"
    required: bool = True
    help_text: str = ""
    # Finding-007 (QA pass, 2026-09-11): a field's value being correctly
    # threaded into the LLM prompt (`LegalDraftEngine._render_fields`) does
    # NOT guarantee the model actually restates it in the generated prose --
    # confirmed live: a `claim_amount` of 65,000, present in the prompt as
    # "Claim Amount (Rs.), if applicable: 65,000", never appeared anywhere
    # in the rendered Facts/Prayer text across several regenerations. For a
    # field this material (the sum a legal notice is actually demanding),
    # that is a real draft-correctness defect, not a stylistic nicety.
    # `True` opts a field into a deterministic post-generation check/patch
    # (`LegalDraftEngine._ensure_verbatim_fields`): if the model didn't
    # state the value, the engine appends one plain, fact-only sentence
    # quoting it verbatim -- never inventing wording the model didn't
    # already imply, just guaranteeing the number itself isn't silently
    # dropped regardless of what the LLM chose to do.
    must_appear_verbatim: bool = False


@dataclass(frozen=True)
class DraftTemplateDefinition:
    draft_id: str
    name: str
    hindi_name: str
    category: str
    description: str
    authority_label: str
    applicable_acts_hint: list[str]
    applicable_sections_hint: list[str]
    required_fields: list[DraftField]
    optional_fields: list[DraftField] = field(default_factory=list)
    drafting_notes: str = ""
    # Natural-language cues (English/Hindi/Hinglish) that identify this draft
    # type from a free-text chat message. Drives `DraftIntentDetector` — kept
    # here (data), not a separate Python rule table.
    trigger_phrases: list[str] = field(default_factory=list)
    # Maps a spoken/typed synonym (lowercase) to a field key, for interpreting
    # in-chat edit commands like "change the police station" or "update my
    # address" against this template's specific fields.
    field_synonyms: dict[str, str] = field(default_factory=dict)
    # Optional `str.format`-style template for the deterministic (non-LLM)
    # fallback's one-line "Subject" — e.g. "Demand for recovery of Rs.
    # {principal_amount}". Referenced field keys are filled from the
    # request's `fields` dict; any key missing/blank at render time falls
    # back to `expected_relief`/`information_sought`/`description` instead
    # (see `LegalDraftEngine._subject_line`), so this is safe to leave unset.
    subject_template: str = ""
    # Part 56 "Advocate-Style Draft Redesign": short, category-specific legal
    # concepts (e.g. "digital fraud", "banking regulations", "electronic
    # evidence" for a cyber crime complaint) that the LLM drafting prompt is
    # instructed to weave naturally into the document's prose where the
    # given facts actually support them -- never as a bolted-on checklist,
    # never forced when a hint doesn't fit. Purely a prompt-generation input;
    # not rendered anywhere directly.
    legal_enrichment_hints: list[str] = field(default_factory=list)

    # Advocate-register redesign: documents a filing of this TYPE ordinarily
    # requires (e.g. RC + insurance + Aadhaar copy for a vehicle-theft
    # complaint), used ONLY as a hedged suggestion ("documents ordinarily
    # required include...") when the user has not confirmed any
    # `available_documents` of their own -- never rendered as a claim that
    # these are actually attached. See `advocate_register.enclosures_block`.
    # Purely advisory data; empty by default, authored per-template only
    # where it adds real value.
    typical_supporting_documents: list[str] = field(default_factory=list)
    # Every Complaint is a formal petition to a police station/consumer forum,
    # so the sworn "I, X, S/o Y, aged Z, residing at..." register always fits
    # -- but "Application" spans BOTH formal petitions to a government or
    # institutional authority (RTI, a certificate, a ration card, a bank NOC)
    # AND ordinary personal correspondence to an employer/school (a
    # resignation letter, a job application, a leave request). The sworn
    # register reads right for the former and wrong for the latter -- nobody
    # opens a resignation letter with "I, Amit Kumar, S/o Rajesh Kumar, aged
    # 28 years, residing at..., do hereby state as under:-". Defaults to
    # True (every Complaint template, and most Applications, want it); the
    # ordinary-correspondence Application templates set this to `false`.
    # Ignored entirely for Notice/Affidavit/Contract, which never use the
    # advocate register regardless of this flag (see `advocate_register.py`).
    advocate_register: bool = True

    # Post-Phase-3 hardening (Phase 2, milestone C).
    #
    # `particulars_fields` -- the field keys this document type renders as its
    # own labelled "Particulars of the matter" block, in this order. Before
    # this existed, `_deterministic_body_sections` had ONE hardcoded list
    # (police station, incident location/date/time, IMEI, accused, witnesses)
    # shared by all 55 templates. Every template was therefore rendered as a
    # police complaint: a consumer complaint's purchase date, amount paid and
    # invoice number appeared nowhere as particulars, and a rent notice's
    # tenancy dates, deposit amount and deductions claimed appeared nowhere
    # either. The values were collected and then silently dropped from the
    # document. Being per-template data rather than Python, a new template
    # gets its own particulars by adding a YAML key.
    particulars_fields: list[str] = field(default_factory=list)
    # A reply/payment period this document type may state WITHOUT the user
    # having chosen one, and the provision that fixes it. Both or neither:
    # `loader._build_template` refuses a period with no basis recorded, because
    # a number with no authority behind it is precisely the arbitrary deadline
    # this pass removed. Where this is unset and the user supplied no
    # `response_deadline_days`, the draft states no period at all rather than
    # falling back to a made-up fifteen days.
    statutory_response_period_days: int | None = None
    statutory_response_period_basis: str = ""
    # Keys from `app/drafting/prohibited_clauses.py` this document type is
    # allowed to state, because the law governing it requires the statement.
    # Declared per template, with the basis in `statutory_response_period_basis`
    # or `drafting_notes`; never inferred.
    permitted_statements: list[str] = field(default_factory=list)

    # --- Problem-first discovery metadata (all optional, all backward
    # compatible -- a YAML file that sets none of these still loads with the
    # defaults below; see `app/drafting/templates/loader.py`). Purely
    # descriptive data consumed by `app/drafting/recommendation.py` and the
    # discovery conversation stages in `app/drafting/conversation.py`; never
    # read by the rendering engine, so getting one of these wrong on an
    # existing template cannot change the document it produces.
    #
    # Broader structural family than `category` -- see `DOCUMENT_FAMILIES`
    # above. Empty string means "not yet classified".
    document_family: str = ""
    # Legal-domain classification (e.g. "property", "financial", "criminal",
    # "consumer", "family", "business"). Free vocabulary -- deliberately not
    # validated against a fixed list, since new domains are expected to be
    # added template-by-template without touching Python.
    domain: str = ""
    subcategory: str = ""
    supported_issues: list[str] = field(default_factory=list)
    user_roles: list[str] = field(default_factory=list)
    opposite_party_roles: list[str] = field(default_factory=list)
    desired_reliefs: list[str] = field(default_factory=list)
    # Case stages (from `CASE_STAGES` above) this template is normally used
    # at. Empty means "not yet classified" -- `recommendation.py` treats an
    # empty list as "no stage signal" rather than "matches no stage".
    case_stages: list[str] = field(default_factory=list)
    forums: list[str] = field(default_factory=list)
    # e.g. {"country": "IN", "level": "central", "states": ["all"]}.
    jurisdiction: dict[str, Any] = field(
        default_factory=lambda: {"country": "IN", "level": "central", "states": ["all"]}
    )
    # Natural-language names for this document keyed by language
    # (english/hindi/hinglish/...), used for search/recommendation ranking.
    # Distinct from `trigger_phrases` (which drives `DraftIntentDetector`'s
    # direct-match scoring) -- aliases are a smaller, curated "what would a
    # person call this" list surfaced in search results and recommendation
    # reasons, not tuned for regex-style detection.
    aliases: dict[str, list[str]] = field(default_factory=dict)
    # Documents ordinarily useful as evidence for this matter, shown to the
    # user as an optional-upload prompt during discovery. Distinct from the
    # older `typical_supporting_documents` (kept unchanged above, used as a
    # hedged in-document suggestion) -- this is the discovery-flow-facing
    # list; template authors may set either or both.
    supporting_documents: list[str] = field(default_factory=list)
    review_requirements: dict[str, bool] = field(
        default_factory=lambda: {
            "jurisdiction_check": False,
            "limitation_check": False,
            "court_fee_check": False,
            "advocate_review_recommended": False,
        }
    )
    template_status: str = "production"
    template_version: str = "1.0"

    def all_fields(self) -> list[DraftField]:
        return [*self.required_fields, *self.optional_fields]

    def field_keys(self) -> set[str]:
        return {draft_field.key for draft_field in self.all_fields()}

    def required_field_keys(self) -> set[str]:
        return {draft_field.key for draft_field in self.required_fields}

    def get_field(self, key: str) -> DraftField | None:
        for draft_field in self.all_fields():
            if draft_field.key == key:
                return draft_field
        return None
