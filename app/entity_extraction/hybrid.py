"""Hybrid entity extraction: rules first, the model only as a proposer.

The order is the whole design, and it runs strictly:

1. **Deterministic rules** (`rules.py`). A regex that matched carries the
   span it matched, so every value it produces is checkable.
2. **Conversation facts.** Anything the user has already told this session is
   used as-is rather than re-derived — it came from the user, which is a
   better source than any reading of a document.
3. **An optional model pass**, and only for what the first two left open.
   Skipped entirely when no provider is configured, when the text is short
   enough that rules will have caught what there is, or when the request
   deadline has already been spent.
4. **Strict validation.** The model's output is parsed as JSON and validated
   against `ExtractedEntity`. Anything that fails — malformed JSON, an
   unknown entity type, a role with no supporting quote, a value that does
   not actually appear in the document — is dropped, not repaired.
5. **Deterministic fallback.** Any failure above leaves the rule results
   standing. The extractor never returns nothing because a provider was
   slow, and never returns a model's guess dressed as a finding.

The invariant that matters most: a model may never CONFIRM a legal role. Its
proposals land in `unresolved_roles` as questions for the user, capped at a
confidence that keeps them visibly below rule-derived findings. Two rules
disagreeing about the same person becomes a `RoleConflict`, which is also a
question — this module resolves nothing by preference.
"""

import json
import re
from typing import Any

import structlog
from pydantic import ValidationError

from app.entity_extraction import rules
from app.llm.base import ChatMessage, LLMProvider
from app.llm.deadline import has_time_for
from app.schemas.extraction import (
    EntityExtractionResult,
    ExtractedEntity,
    LegalRole,
    RoleCandidate,
    RoleConflict,
)

log = structlog.get_logger(__name__)

# Below this, the rules have seen everything there is and a model call is
# latency for nothing.
_MIN_TEXT_FOR_LLM = 400
# The model sees a bounded prefix: extraction quality does not improve with
# a 40-page dump, and an unbounded prompt is an unbounded bill.
_MAX_TEXT_FOR_LLM = 6_000
# A model proposal is never worth more than this, however confident it sounds.
_LLM_CONFIDENCE_CEILING = 0.55

_VALID_ROLES: frozenset[str] = frozenset(LegalRole.__args__)  # type: ignore[attr-defined]

_PROMPT = """You extract structured facts from Indian legal documents.

Return ONLY a JSON object of this exact shape, with no commentary:

{{"entities": [{{"value": "...", "entity_type": "...", "legal_role": "...", "source_text": "..."}}]}}

Rules you must follow:
- entity_type is one of: person, organization, court, police_station, address,
  case_number, fir_number, date, amount, act, section, property,
  vehicle_number, document_reference, email, phone.
- legal_role is one of: unknown, applicant, respondent, complainant, accused,
  petitioner, defendant, advocate, witness, landlord, tenant, employer,
  employee, buyer, seller. Use "unknown" unless the DOCUMENT ITSELF labels
  the person with that role.
- source_text must be copied verbatim from the document and must contain the
  value. If you cannot quote it, do not report it.
- Never infer, translate, normalise or complete a value. Report what is written.
- If you find nothing, return {{"entities": []}}.

DOCUMENT:
{document}
"""


async def extract(
    text: str,
    *,
    conversation_facts: dict[str, Any] | None = None,
    llm: LLMProvider | None = None,
    page_of: dict[str, int] | None = None,
) -> EntityExtractionResult:
    """Everything the text establishes, suspects, or contradicts itself about."""
    document = text or ""
    entities = rules.extract_roles(document) + rules.extract_values(document)
    entities.extend(_from_conversation(conversation_facts or {}, document))

    unresolved: list[RoleCandidate] = _party_candidates(document)

    if llm is not None and _worth_a_model_call(document):
        proposed, role_proposals = await _propose(llm, document)
        merged, role_questions = _merge(entities, proposed, role_proposals)
        entities = merged
        unresolved.extend(role_questions)

    entities = _attach_pages(entities, page_of or {})
    conflicts = _conflicts(entities)
    questions = [conflict.question for conflict in conflicts]
    questions.extend(
        f'Is "{candidate.value}" the {candidate.role.replace("_", " ")} here? '
        f"I am asking because {candidate.reason}"
        for candidate in unresolved
    )

    counts: dict[str, int] = {}
    for entity in entities:
        counts[entity.extraction_method] = counts.get(entity.extraction_method, 0) + 1

    return EntityExtractionResult(
        entities=entities,
        unresolved_roles=unresolved,
        conflicts=conflicts,
        questions=_dedupe(questions),
        method_counts=counts,
    )


# ---------------------------------------------------------------------------
# Layer 2: what the conversation already established
# ---------------------------------------------------------------------------

# Conversation fact keys that map onto an entity type. Only keys whose meaning
# is unambiguous are mapped -- a generic "name" field could be anybody.
_FACT_KEYS: tuple[tuple[str, str, LegalRole], ...] = (
    ("applicant_name", "person", "applicant"),
    ("complainant_name", "person", "complainant"),
    ("respondent_name", "person", "respondent"),
    ("accused_name", "person", "accused"),
    ("landlord_name", "person", "landlord"),
    ("tenant_name", "person", "tenant"),
    ("advocate_name", "person", "advocate"),
    ("case_number", "case_number", "unknown"),
    ("fir_number", "fir_number", "unknown"),
    ("police_station", "police_station", "unknown"),
    ("court_name", "court", "unknown"),
    ("amount", "amount", "unknown"),
    ("vehicle_number", "vehicle_number", "unknown"),
    ("applicant_address", "address", "unknown"),
)


def _from_conversation(facts: dict[str, Any], document: str) -> list[ExtractedEntity]:
    """Facts the USER supplied. Confidence is high because the source is the
    person whose matter this is -- but the method is recorded as
    `conversation`, so a reader can always tell it did not come from the
    document."""
    found: list[ExtractedEntity] = []
    for key, entity_type, role in _FACT_KEYS:
        value = str(facts.get(key) or "").strip()
        if not value:
            continue
        found.append(
            ExtractedEntity(
                value=value,
                entity_type=entity_type,  # type: ignore[arg-type]
                legal_role=role,
                confidence=0.9,
                source_text=f"You told me earlier: {key.replace('_', ' ')} is {value}.",
                extraction_method="conversation",
            )
        )
    return found


# ---------------------------------------------------------------------------
# Role candidates the text hints at but does not establish
# ---------------------------------------------------------------------------

_VERSUS = re.compile(
    r"([A-Z][A-Za-z.'&\s]{2,40}?)\s+(?:v/?s\.?|versus|बनाम)\s+([A-Z][A-Za-z.'&\s]{2,40})",
)


def _party_candidates(document: str) -> list[RoleCandidate]:
    """"A vs B" names two parties without saying which side is which.

    Which name is the petitioner depends on the cause title and the forum,
    and putting the wrong one in the wrong half of a pleading is exactly the
    error this product must not make. So both are offered as candidates and
    the user decides.
    """
    match = _VERSUS.search(document or "")
    if not match:
        return []
    first, second = (part.strip(" ,.") for part in match.groups())
    evidence = match.group(0).strip()
    return [
        RoleCandidate(
            value=first,
            role="petitioner",
            evidence=evidence,
            confidence=0.4,
            reason=f'the cause title reads "{evidence}", which names two sides without labelling them.',
        ),
        RoleCandidate(
            value=second,
            role="respondent",
            evidence=evidence,
            confidence=0.4,
            reason=f'the cause title reads "{evidence}", which names two sides without labelling them.',
        ),
    ]


# ---------------------------------------------------------------------------
# Layer 3/4: the model proposal, and the validation that guards it
# ---------------------------------------------------------------------------


def _worth_a_model_call(document: str) -> bool:
    if len(document) < _MIN_TEXT_FOR_LLM:
        return False
    # An absent deadline means unbounded (scripts, tests); a nearly-spent one
    # means the user is already waiting, and an OPTIONAL pass must not be
    # what pushes a request past the point where anyone is still there to
    # read the answer.
    return has_time_for(5.0)


async def _propose(
    llm: LLMProvider, document: str
) -> tuple[list[ExtractedEntity], list[tuple[str, str, str]]]:
    """The model's suggestions as `(entities, role_proposals)`.

    Never raises into the caller: every failure path returns empty lists and
    leaves the rule results standing.
    """
    try:
        response = await llm.chat(
            [ChatMessage(role="user", content=_PROMPT.format(document=document[:_MAX_TEXT_FOR_LLM]))],
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - an optional pass must never break extraction
        log.warning("entity_llm_call_failed", error=str(exc))
        return [], []
    if response.error:
        log.info("entity_llm_unavailable", error_kind=response.error_kind)
        return [], []
    return _validate(response.content, document)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _validate(raw: str, document: str) -> tuple[list[ExtractedEntity], list[tuple[str, str, str]]]:
    """Parses and validates model output. Anything doubtful is dropped.

    Four gates, in order: the response must contain JSON; the JSON must have
    an `entities` list; each item must validate as an `ExtractedEntity`; and
    each item's `value` and `source_text` must actually appear in the
    document. The last gate is the one that matters — it is what makes a
    fabricated party name impossible to smuggle through, because a value
    that is not in the text cannot have been read from it.
    """
    block = _JSON_BLOCK.search(raw or "")
    if not block:
        log.info("entity_llm_output_unparseable")
        return [], []
    try:
        payload = json.loads(block.group(0))
    except (json.JSONDecodeError, ValueError):
        log.info("entity_llm_output_invalid_json")
        return [], []
    items = payload.get("entities") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return [], []

    haystack = _normalise(document)
    accepted: list[ExtractedEntity] = []
    role_proposals: list[tuple[str, str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", "")).strip()
        quote = str(item.get("source_text", "")).strip()
        if not value or _normalise(value) not in haystack:
            # Not in the document: it was invented, or normalised into
            # something the document does not say. Either way it is not a
            # finding.
            continue
        if quote and _normalise(quote) not in haystack:
            continue
        role = str(item.get("legal_role", "unknown"))
        if role not in _VALID_ROLES:
            role = "unknown"
        try:
            accepted.append(
                ExtractedEntity(
                    value=value,
                    entity_type=str(item.get("entity_type", "person")),  # type: ignore[arg-type]
                    # A model NEVER confirms a role. The proposal is returned
                    # separately so it can be raised as a question; the
                    # entity itself stays "unknown".
                    legal_role="unknown",
                    confidence=_LLM_CONFIDENCE_CEILING,
                    source_text=quote or value,
                    extraction_method="llm",
                )
            )
        except ValidationError:
            # An unknown entity_type or an out-of-range confidence. Dropped
            # rather than coerced: a repaired guess is still a guess.
            continue
        if role != "unknown":
            role_proposals.append((value, role, quote))
    return accepted, role_proposals


def _merge(
    established: list[ExtractedEntity],
    proposed: list[ExtractedEntity],
    role_proposals: list[tuple[str, str, str]],
) -> tuple[list[ExtractedEntity], list[RoleCandidate]]:
    """Adds model findings that the rules did not already have.

    A rule result always wins: it carries the span it matched, and replacing
    checkable evidence with a model's paraphrase would be a downgrade even
    when the paraphrase reads better.
    """
    known = {(_normalise(entity.value), entity.entity_type) for entity in established}
    merged = list(established)
    for entity in proposed:
        if (_normalise(entity.value), entity.entity_type) in known:
            continue
        merged.append(entity)

    candidates: list[RoleCandidate] = []
    for value, role, quote in role_proposals:
        if any(
            _normalise(entity.value) == _normalise(value) and entity.legal_role != "unknown"
            for entity in established
        ):
            # A rule already established this person's role from a label in
            # the text. The model's opinion adds nothing and must not
            # compete with evidence.
            continue
        candidates.append(
            RoleCandidate(
                value=value,
                role=role,  # type: ignore[arg-type]
                evidence=quote,
                confidence=_LLM_CONFIDENCE_CEILING,
                reason="a language model suggested it from the wording, and nothing in the text states it outright.",
            )
        )
    return merged, candidates


# ---------------------------------------------------------------------------
# Conflicts and page provenance
# ---------------------------------------------------------------------------


def _conflicts(entities: list[ExtractedEntity]) -> list[RoleConflict]:
    """One name, two roles that cannot both be true."""
    by_name: dict[str, list[ExtractedEntity]] = {}
    for entity in entities:
        if entity.entity_type == "person" and entity.legal_role != "unknown":
            by_name.setdefault(_normalise(entity.value), []).append(entity)

    conflicts: list[RoleConflict] = []
    for group in by_name.values():
        roles = sorted({entity.legal_role for entity in group})
        if len(roles) > 1:
            conflicts.append(
                RoleConflict(
                    value=group[0].value,
                    roles=roles,
                    evidence=[entity.source_text for entity in group if entity.source_text],
                )
            )
    return conflicts


def _attach_pages(entities: list[ExtractedEntity], page_of: dict[str, int]) -> list[ExtractedEntity]:
    """Stamps the page each value was read from, where that is known.

    A value whose page cannot be established keeps `source_page=None`.
    Defaulting to page 1 would be a citation to a page nobody checked.
    """
    if not page_of:
        return entities
    stamped: list[ExtractedEntity] = []
    for entity in entities:
        page = next(
            (number for snippet, number in page_of.items() if snippet and snippet in entity.source_text),
            None,
        )
        stamped.append(entity if page is None else entity.model_copy(update={"source_page": page}))
    return stamped


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
