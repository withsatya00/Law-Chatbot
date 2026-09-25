"""One response invariant for the strict-RAG refusal.

The reported defect
-------------------
A turn whose settled answer was "no verified document related to this question
is available in the Knowledge Base" was displayed alongside a Sources block
citing the BNS and a GST circular, an `applicable_law` list, page evidence and
a "how current are these sources?" note. None of it supported the answer --
the answer says there is no support. The refusal reaches the user through
three different routes and each one built its response from whatever
`sources`/`ranked` happened to be in scope at the time:

* `ChatService.answer()` -- retrieval succeeded, but the grounding validator
  or the LLM itself judged the chunks did not answer the question, so the
  answer was swapped for the refusal while `sources`/`ranked` stayed;
* `ChatService.answer_stream()` -- the same, plus the post-stream quality gate;
* `ChatService._respond_from_cache()` -- a refusal stored before this module
  existed, replayed with its stored citations intact.

So the fix is one invariant applied at all three, not three local patches:
if the settled answer IS the refusal, then there is nothing supporting it, and
every field that would suggest otherwise is emptied -- before history is
written, so a misleading citation is never stored either.
"""

import re
from typing import Any

from app.core.constants import is_no_verified_context

# The response fields that assert support for an answer. Every one of them is
# emptied when the answer is a refusal. `answer` and `confidence_reason` are
# deliberately NOT here: the refusal text and the reason it was reached are
# exactly what the user should still see.
SAFE_DECLINE_CLEARED_FIELDS: tuple[str, ...] = (
    "sources",
    "evidence_pages",
    "applicable_law",
    "retrieved_sections",
    "retrieved_chunks",
    "currency_notice",
)

def is_safe_decline(answer: str) -> bool:
    """Whether `answer` is the strict-RAG refusal, in any language."""
    return is_no_verified_context(answer)


def clear_support[S, R](answer: str, sources: list[S], ranked: list[R]) -> tuple[list[S], list[R], float, bool]:
    """`(sources, ranked, confidence, no_verified_context)` for this answer.

    Returns the inputs untouched for a real answer; empty lists, zero
    confidence and `True` for a refusal. Call it BEFORE persisting memory or
    inserting history so the emptied values are what gets stored, and the
    response built from them carries no citation, no page evidence, no
    applicable-law entry and no currency notice.
    """
    if not is_safe_decline(answer):
        return sources, ranked, 0.0, False
    return [], [], 0.0, True


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Defensive sanitization of an already-built response-shaped dict.

    Used for cached entries written before the invariant existed (a stored
    refusal keeps its stored `sources`/`retrieved_sections`), and safe to
    apply to any payload: a dict whose `answer` is not a refusal is returned
    unchanged, and a key the payload does not have is not added.
    """
    if not is_safe_decline(str(payload.get("answer") or "")):
        return payload
    cleaned = dict(payload)
    for name in SAFE_DECLINE_CLEARED_FIELDS:
        if name in cleaned:
            cleaned[name] = "" if isinstance(cleaned[name], str) else []
    if "confidence" in cleaned:
        cleaned["confidence"] = 0.0
    cleaned["no_verified_context"] = True
    return cleaned


_PROVISION_CUE = re.compile(
    r"\b(?:act|sanhita|code|section|sec\.|article|rules?|regulations?|constitution)\b"
    r"|धारा|कलम|अधिनियम|संहिता|ধারা|আইন|ધારા|કલમ|பிரிவு|சட்டம்|సెక్షన్|చట్టం|ಸೆಕ್ಷನ್|ಕಾಯ್ದೆ|വകുപ്പ്|നിയമം|ਧਾਰਾ|ଧାରା|دفعہ|ایکٹ|ڪلم",
    re.IGNORECASE,
)


def _act_key_terms(act_name: str) -> list[str]:
    stop = {"the", "act", "of", "and", "code", "sanhita"}
    return [w for w in re.findall(r"[a-z]{4,}", act_name.lower()) if w not in stop]


def prune_unreferenced_sources[S, R](answer: str, sources: list[S], ranked: list[R]) -> tuple[list[S], list[R]]:
    """Drops cited Acts the answer never mentions, when other sources ARE mentioned.

    Retrieval sometimes drags in an unrelated Act (confirmed live: two
    "Bombay Gas Supply Act" sections cited under a defective-mobile consumer
    answer). Conservative on purpose: nothing is pruned unless at least one
    source with a named Act is actually referenced in the answer text, a
    source with no Act name is always kept, and at least one source survives.
    """
    text = (answer or "").lower()

    def referenced(source: S) -> bool | None:
        act = getattr(source, "act_name", None)
        if not act:
            return None
        terms = _act_key_terms(str(act))
        return bool(terms) and all(term in text for term in terms)

    verdicts = [referenced(source) for source in sources]
    if not any(v is True for v in verdicts):
        # The answer references none of its sources. If it also names no
        # Act/section at all (a clarifying question -- confirmed live, a
        # Marathi "which details do you need?" reply carried five unrelated
        # Maharashtra Acts), nothing in it rests on them, so listing them
        # would present unrelated law as its basis.
        asks_questions = ((answer or "").count("?") + (answer or "").count("؟")) >= 2
        if sources and asks_questions and not _PROVISION_CUE.search(answer or ""):
            return [], []
        return sources, ranked
    kept = [s for s, v in zip(sources, verdicts, strict=True) if v is not False]
    dropped_acts = {str(getattr(s, "act_name", "")) for s, v in zip(sources, verdicts, strict=True) if v is False}
    kept_ranked = [
        r for r in ranked
        if str((getattr(r, "metadata", None) or {}).get("act_name") or "") not in dropped_acts
    ]
    return kept, kept_ranked or ranked
