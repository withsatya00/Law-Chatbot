"""Deterministic, LLM-free draft recommendation engine.

Scores every template against a structured `MatterProfile`-shaped dict (see
`app/drafting/discovery.py` for the conversational flow that builds one) and
returns at most three ranked candidates. Pure function of its inputs -- same
profile, same template catalogue, same result every time -- so it is testable
without mocking an LLM and safe to run on every discovery turn.

Scoring is a fixed points table (module docstring intentionally mirrors it so
the two never drift):

- Exact template/alias match:      +40
- Supported-issue match (each):    +20
- Desired-relief match (each):     +15
- User-role match:                 +10
- Case-stage match:                +10
- Domain/subcategory match:        +5 each
- Contradictory role (the profile's role is this template's OPPOSITE party's
  role, e.g. profile says "tenant" but the template is landlord-only): the
  template is excluded outright, not merely penalised -- recommending a
  landlord-only eviction notice to a tenant is worse than recommending
  nothing.
- Case-stage mismatch (template declares stages but not the profile's one):
  -20, since a document type still commonly fits at a neighbouring stage.

The raw score is normalised to a 0-1 confidence by dividing by
`_CONFIDENCE_DIVISOR` (chosen so a single exact alias match alone lands at the
top of the "high confidence" band -- see module-level comment above the
constant). Confidence tiers drive how the discovery conversation presents
results (see `RecommendationTier`); this module only computes them.
"""

import re
from dataclasses import dataclass
from typing import Literal

from app.drafting.intent import _typo_tolerant_match
from app.drafting.templates import list_templates
from app.drafting.templates.base import DraftTemplateDefinition

RecommendationTier = Literal["high", "medium", "low", "none"]

_MAX_CANDIDATES = 3

# A single exact alias/template-name match (+40) should already read as
# "high confidence" (>=0.85) on its own, since it is the strongest possible
# signal short of the user typing the literal template name (which never
# reaches this engine at all -- `DraftIntentDetector` intercepts that
# earlier). 40 / 47 = 0.851, just over the high-confidence threshold, while a
# full non-alias match (issue+relief+role+stage+domain = 20+15+10+10+5 = 60)
# clips to 1.0 confidence.
_CONFIDENCE_DIVISOR = 47.0

_SCORE_ALIAS_MATCH = 40
_SCORE_ISSUE_MATCH = 20
_SCORE_RELIEF_MATCH = 15
_SCORE_ROLE_MATCH = 10
_SCORE_STAGE_MATCH = 10
_SCORE_DOMAIN_MATCH = 5
_SCORE_SUBCATEGORY_MATCH = 5
_PENALTY_STAGE_MISMATCH = -20
_PENALTY_ROLE_MISMATCH = -30

HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.60


@dataclass(frozen=True)
class RecommendationCandidate:
    draft_id: str
    name: str
    hindi_name: str
    score: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class RecommendationResult:
    tier: RecommendationTier
    candidates: tuple[RecommendationCandidate, ...] = ()
    # Set only when `tier == "none"` -- a human-readable explanation that no
    # template matched, for the "clarify / browse category / generic draft"
    # fallback reply. Never a guess at what the user might have meant.
    no_match_reason: str = ""


def _profile_get(profile: dict[str, object], key: str, default: object = None) -> object:
    value = profile.get(key)
    return value if value is not None else default


def _normalize_text_list(values: object) -> set[str]:
    """`values` comes straight from a caller-supplied profile dict (see
    `_profile_get`), so it's untrusted shape, not just untrusted content.
    Before this guard, a bare string here (e.g. `"issues": "bail"` instead
    of `["bail"]`) would silently iterate character-by-character -- `str` is
    iterable -- producing `{"b", "a", "i", "l"}` instead of `{"bail"}"`, a
    wrong-but-non-crashing result that would never surface as an error. Any
    other non-iterable (an int, a bool) would raise `TypeError` outright.
    """
    if not values:
        return set()
    if isinstance(values, str):
        text = values.strip().lower()
        return {text} if text else set()
    if not isinstance(values, (list, tuple, set, frozenset)):
        return set()
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _alias_texts(template: DraftTemplateDefinition) -> set[str]:
    texts = {template.name.lower(), template.hindi_name.lower(), template.draft_id.lower().replace("_", " ")}
    for names in template.aliases.values():
        texts.update(name.lower() for name in names)
    return texts


def _score_template(
    template: DraftTemplateDefinition, profile: dict[str, object], restrict_ids: frozenset[str] | None
) -> tuple[float, list[str]] | None:
    """Returns `(score, reasons)`, or `None` if this template is excluded
    outright (a contradictory role, or not in `restrict_ids` when given)."""
    if restrict_ids is not None and template.draft_id not in restrict_ids:
        return None

    user_role = str(_profile_get(profile, "user_role") or "").strip().lower()
    if (
        user_role
        and template.user_roles
        and user_role not in {r.lower() for r in template.user_roles}
        and user_role in {r.lower() for r in template.opposite_party_roles}
    ):
        # The profile's role is explicitly this template's OTHER party's
        # role -- e.g. a tenant asking for a landlord-only eviction notice.
        # Excluded, never merely penalised.
        return None

    score = 0.0
    reasons: list[str] = []

    raw_text = str(_profile_get(profile, "raw_description") or "").strip().lower()
    if raw_text and raw_text in _alias_texts(template):
        score += _SCORE_ALIAS_MATCH
        reasons.append("exact match with how you described the document")

    issues = _normalize_text_list(_profile_get(profile, "issues"))
    matched_issues = issues & {i.lower() for i in template.supported_issues}
    if matched_issues:
        score += _SCORE_ISSUE_MATCH * len(matched_issues)
        reasons.append("matches the issue: " + ", ".join(sorted(matched_issues)))

    reliefs = _normalize_text_list(_profile_get(profile, "desired_reliefs"))
    matched_reliefs = reliefs & {r.lower() for r in template.desired_reliefs}
    if matched_reliefs:
        score += _SCORE_RELIEF_MATCH * len(matched_reliefs)
        reasons.append("matches the relief you want: " + ", ".join(sorted(matched_reliefs)))

    if user_role and template.user_roles:
        if user_role in {r.lower() for r in template.user_roles}:
            score += _SCORE_ROLE_MATCH
            reasons.append(f"matches your role ({user_role})")
        else:
            score += _PENALTY_ROLE_MISMATCH

    case_stage = str(_profile_get(profile, "case_stage") or "").strip().lower()
    if case_stage and template.case_stages:
        if case_stage in {s.lower() for s in template.case_stages}:
            score += _SCORE_STAGE_MATCH
            reasons.append(f"matches your matter stage ({case_stage})")
        else:
            score += _PENALTY_STAGE_MISMATCH

    domain = str(_profile_get(profile, "domain") or "").strip().lower()
    if domain and template.domain and domain == template.domain.lower():
        score += _SCORE_DOMAIN_MATCH
        reasons.append(f"matches the domain ({domain})")

    subcategory = str(_profile_get(profile, "subcategory") or "").strip().lower()
    if subcategory and template.subcategory and subcategory == template.subcategory.lower():
        score += _SCORE_SUBCATEGORY_MATCH

    return score, reasons


_SEARCH_DEFAULT_LIMIT = 5


@dataclass(frozen=True)
class SearchResult:
    draft_id: str
    name: str
    hindi_name: str
    category: str
    domain: str
    score: float


def search_templates(
    query: str, *, limit: int = _SEARCH_DEFAULT_LIMIT, templates: list[DraftTemplateDefinition] | None = None
) -> list[SearchResult]:
    """Deterministic, ranked, LIMITED free-text search over the template
    catalogue -- backs `GET /draft-templates/search` and the frontend's
    "Search a draft" entry action. Never returns more than `limit` results
    (default 5, per the "search results must be limited and ranked" rule) --
    this is a search box, not a substitute for the full catalogue dump the
    rest of this feature deliberately avoids showing.
    """
    normalized_query = (query or "").strip().lower()
    if not normalized_query:
        return []
    query_tokens = {t for t in normalized_query.split() if t}
    scored: list[tuple[float, SearchResult]] = []
    for template in templates or list_templates():
        haystacks = _alias_texts(template) | {template.category.lower(), template.domain.lower(), template.subcategory.lower()}
        haystacks |= {phrase.lower() for phrase in template.trigger_phrases}
        best = 0.0
        for text in haystacks:
            if not text:
                continue
            if text == normalized_query:
                best = max(best, 100.0)
            # Word-boundary-aware, not a bare Python `in` substring check --
            # "rti" is a real, common abbreviation (Right to Information)
            # that is ALSO a literal substring of the unrelated word
            # "certificate" ("ce-RTI-ficate"). A plain substring test made
            # every certificate-application template rank for a search of
            # "rti" purely by typographic accident. `\b...\b` only matches
            # "rti" as its own word/token, not embedded inside a longer one.
            elif re.search(rf"\b{re.escape(normalized_query)}\b", text) or re.search(
                rf"\b{re.escape(text)}\b", normalized_query
            ):
                best = max(best, 60.0)
            else:
                text_tokens = set(text.split())
                overlap = query_tokens & text_tokens
                if overlap:
                    best = max(best, 20.0 * len(overlap))
                else:
                    # Typo-tolerant fallback: "renta agrement" (no exact
                    # token overlap with "rent agreement" at all -- neither
                    # word is spelled correctly) previously scored 0 and
                    # returned no results for a plainly-intended, one-typo
                    # search. Scored lower than a genuine exact-token
                    # overlap (15 vs 20 per word) since it's weaker evidence.
                    fuzzy_overlap = sum(
                        1 for qt in query_tokens
                        if any(_typo_tolerant_match(qt, tt) for tt in text_tokens)
                    )
                    if fuzzy_overlap:
                        best = max(best, 15.0 * fuzzy_overlap)
        if best > 0:
            scored.append((
                best,
                SearchResult(
                    draft_id=template.draft_id, name=template.name, hindi_name=template.hindi_name,
                    category=template.category, domain=template.domain, score=best,
                ),
            ))
    scored.sort(key=lambda item: (-item[0], item[1].draft_id))
    return [result for _score, result in scored[:limit]]


def recommend(
    profile: dict[str, object],
    *,
    restrict_ids: "frozenset[str] | None" = None,
    templates: list[DraftTemplateDefinition] | None = None,
) -> RecommendationResult:
    """Scores every template (or only those in `restrict_ids`, when given)
    against `profile` and returns at most three ranked candidates.

    `profile` is the plain-dict shape produced by
    `app.drafting.discovery.default_matter_profile()` -- kept as a dict
    rather than a dataclass/pydantic model so callers can pass the exact
    object stored in conversation memory without a conversion step.
    """
    candidates: list[tuple[float, RecommendationCandidate]] = []
    for template in templates or list_templates():
        scored = _score_template(template, profile, restrict_ids)
        if scored is None:
            continue
        score, reasons = scored
        if score <= 0:
            continue
        confidence = min(1.0, max(0.0, score) / _CONFIDENCE_DIVISOR)
        reason = "; ".join(reasons) if reasons else "closest available match to what you described"
        candidates.append((
            score,
            RecommendationCandidate(
                draft_id=template.draft_id,
                name=template.name,
                hindi_name=template.hindi_name,
                score=score,
                confidence=round(confidence, 3),
                reason=reason,
            ),
        ))

    if not candidates:
        return RecommendationResult(
            tier="none",
            no_match_reason=(
                "No template in the catalogue matches the details given so far. "
                "This does not mean nothing can be drafted -- it means none of the "
                "existing templates precisely fit yet."
            ),
        )

    # Deterministic, stable ordering: score desc, then draft_id asc so equal
    # scores never depend on dict/set iteration order.
    candidates.sort(key=lambda item: (-item[0], item[1].draft_id))
    top = [candidate for _score, candidate in candidates[:_MAX_CANDIDATES]]
    top_confidence = top[0].confidence
    if top_confidence >= HIGH_CONFIDENCE_THRESHOLD:
        tier: RecommendationTier = "high"
        top = top[:1]
    elif top_confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        tier = "medium"
    else:
        tier = "low"
    return RecommendationResult(tier=tier, candidates=tuple(top))
