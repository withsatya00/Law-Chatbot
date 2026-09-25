"""Phase 1 "Multi-Intent Workflow Orchestration": recognizes and executes
ONLY the explicitly allowlisted two-step intent chains below.

Never executes an arbitrary LLM-proposed sequence of intents -- `ALLOWED_
CHAINS` is the sole source of truth for what a "workflow" is in this
product. `detect_chain` is a pure function over the classifier's own
`detected_intents` (already computed by `ConversationIntentClassifier.
classify_advanced`, no extra LLM call), and `map_facts_to_draft_fields` is
a pure, conservative fact-mapper with no I/O -- both independently unit
testable without touching `ChatService`. The actual cross-step execution
(document ownership re-verification, calling `DocumentService.analyze`,
seeding `DraftConversationEngine`) lives in `ChatService` itself, since it
needs those collaborators.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.drafting.templates.base import DraftTemplateDefinition
    from app.schemas.document import DocumentAnalysisResponse

# Intents that fall through to the general RAG/answer path rather than a
# dedicated handler (see `ChatService._dispatch_conversation_intent`) --
# treated as a single "Legal Research" step for chain-matching purposes,
# since the user's spec names one "Legal Research" step but the classifier
# has no single intent by that name.
RAG_INTENTS = frozenset({
    "Legal Explanation", "Legal Advice", "General Legal Information",
    "Legal Procedure", "Law Comparison", "Legal Dictionary",
})

_RAG_MARKER = "_LEGAL_RESEARCH"

# Ordered (step1, step2) pairs -- order is the actual execution order, not
# the order the user's phrasing or the classifier's primary/secondary
# ranking implies. A draft can't be grounded in facts that haven't been
# extracted yet, so "Document Analysis" always executes before "Draft
# Generation" regardless of which one the classifier called primary.
ALLOWED_CHAINS: tuple[tuple[str, str], ...] = (
    ("Document Analysis", "Draft Generation"),
    ("Document Analysis", _RAG_MARKER),
    (_RAG_MARKER, "Draft Generation"),
    ("Translation", "Response Modification"),
)
MAX_CHAIN_LENGTH = 2


def _normalize(intent: str) -> str:
    return _RAG_MARKER if intent in RAG_INTENTS else intent


def detect_chain(detected_intents: tuple[str, ...]) -> tuple[str, str] | None:
    """Returns the matching allowed chain, in its canonical execution
    order, if `detected_intents` covers both steps of exactly one
    allowlisted chain -- else `None`.

    `detected_intents` is unordered w.r.t. execution (it's the
    classifier's confidence-sorted candidate set); the returned pair is
    always in `ALLOWED_CHAINS`' own fixed order, resolving the internal
    `_LEGAL_RESEARCH` marker back to whichever real RAG-family intent was
    actually detected.
    """
    if len(detected_intents) < 2:
        return None
    normalized = {_normalize(intent) for intent in detected_intents}
    for step1, step2 in ALLOWED_CHAINS:
        if step1 not in normalized or step2 not in normalized:
            continue
        actual_step1 = step1 if step1 != _RAG_MARKER else next(i for i in detected_intents if i in RAG_INTENTS)
        actual_step2 = step2 if step2 != _RAG_MARKER else next(i for i in detected_intents if i in RAG_INTENTS)
        if actual_step1 == actual_step2:
            continue
        return actual_step1, actual_step2
    return None


# Field keys, across the various draft template YAMLs, that plausibly hold
# a document's own narrative/summary content -- checked in order, first
# match wins, so a template is never given two conflicting seed values.
_NARRATIVE_FIELD_KEYS = ("facts", "description", "incident_details", "complaint_details", "grounds")
# Field keys that represent an unambiguous single reference date -- never a
# deadline/response-window field, since a date copied verbatim from the
# source document is not the same thing as a deadline the user intends to
# set going forward.
_REFERENCE_DATE_FIELD_KEYS = ("incident_date", "date_of_incident", "agreement_date", "date_of_agreement")


def map_facts_to_draft_fields(
    template: "DraftTemplateDefinition", analysis: "DocumentAnalysisResponse"
) -> dict[str, str]:
    """Conservative, deterministic subset mapping from a document-analysis
    result into a draft template's own field keys.

    Never fabricates a value and never guesses between multiple
    candidates for the same role -- e.g. a document naming two people has
    no reliable way to know which is the applicant vs. the respondent
    without role-classified entity extraction (which doesn't exist yet,
    see the feature checklist), so person names are deliberately NOT
    mapped here. Anything left unmapped simply falls through to
    `DraftConversationEngine`'s existing "ask for what's still missing"
    flow, which is the safe behavior, not a gap this function needs to
    close by guessing.
    """
    mapped: dict[str, str] = {}
    entities = (analysis.structured_data or {}).get("entities") or {}
    field_keys = template.field_keys()

    summary = (analysis.executive_summary or "").strip()
    if summary:
        for key in _NARRATIVE_FIELD_KEYS:
            if key in field_keys:
                mapped[key] = summary
                break

    amounts = entities.get("amount") or []
    if len(amounts) == 1:
        for key in field_keys:
            if "amount" in key:
                mapped[key] = amounts[0]

    case_numbers = entities.get("case_number") or []
    if len(case_numbers) == 1:
        for key in field_keys:
            if "case_number" in key or "fir_number" in key:
                mapped[key] = case_numbers[0]

    dates = entities.get("date") or []
    if len(dates) == 1:
        for key in _REFERENCE_DATE_FIELD_KEYS:
            if key in field_keys:
                mapped[key] = dates[0]
                break

    return mapped
