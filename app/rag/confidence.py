"""Separate confidence dimensions for a grounded legal answer.

A single blended number cannot distinguish the two failure modes that matter
most here. "The model wrote a fluent answer from thin sources" and "the model
hedged over excellent sources" both land mid-scale, and a reader cannot tell
which they are looking at. Worse, a high blended score on an ungrounded answer
is an invitation to act on it.

So three inputs are reported separately and one derived number is reported for
backward compatibility:

  * `retrieval_confidence`   -- did we find plausibly relevant material at all?
                               From the reranked chunk scores.
  * `source_grounding_confidence`
                            -- is the answer actually supported by what we
                               found, and is what we found identifiable law?
                               From citation quality and grounding validation.
  * `model_confidence`       -- how settled is the generator's own output?
                               From intent/entity certainty and whether the
                               quality gate had to intervene.
  * `overall_confidence`     -- the single number existing clients already read.

**Grounding is a ceiling, not a term.** `overall` can never exceed
`source_grounding_confidence`, because an answer that is not supported by
verified retrieved text does not become more trustworthy by being fluent or by
matching a confident intent classification. A blended average would let strong
retrieval and a confident model paper over weak grounding, which is the exact
combination that produces a confident-sounding wrong legal answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Below this, an answer must not be presented as a confident legal statement --
# `ConfidenceBreakdown.is_high` gates on it and callers use that to decide
# whether to hedge. Chosen to sit above the score an answer gets from
# unidentifiable sources alone.
HIGH_CONFIDENCE_FLOOR = 0.55


@dataclass(frozen=True)
class ConfidenceBreakdown:
    retrieval: float
    source_grounding: float
    model: float
    overall: float
    reason: str

    @property
    def is_high(self) -> bool:
        return self.overall >= HIGH_CONFIDENCE_FLOOR

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval_confidence": self.retrieval,
            "source_grounding_confidence": self.source_grounding,
            "model_confidence": self.model,
            "overall_confidence": self.overall,
            "confidence_reason": self.reason,
        }


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 2)


def retrieval_confidence(chunk_scores: list[float]) -> float:
    """How well the retrieved material scored, from the top few chunks.

    Only the top three count: a long tail of weak chunks says nothing about
    whether the question was answerable, and averaging over all of them
    punishes a broad recall pass that found the right chunk first.
    """
    top = [score for score in chunk_scores[:3] if score > 0]
    if not top:
        return 0.0
    return _clamp(sum(top) / len(top))


def source_grounding_confidence(
    *,
    citations: list[Any],
    grounded: bool,
    verified_sources: int = 0,
) -> float:
    """How well the answer is anchored to identifiable, retrieved law.

    Returns 0.0 when grounding validation failed or nothing was cited. There is
    no partial credit for an ungrounded answer: the question this dimension
    answers is "can a reader check this?", and the answer is either yes or no.
    """
    if not grounded or not citations:
        return 0.0
    identifiable = sum(1 for citation in citations if getattr(citation, "is_identifiable", False))
    # A citation naming only an internal filename is not something a reader can
    # look up, so it contributes materially less than one naming an Act and
    # section.
    base = 0.45 + 0.40 * (identifiable / len(citations))
    with_pages = sum(
        1 for citation in citations
        if getattr(citation, "page_number", None) is not None
        or getattr(citation, "page_start", None) is not None
    )
    if with_pages:
        # Page-level evidence makes a citation checkable in the strongest sense
        # available: a reader can open that page.
        base += 0.08 * (with_pages / len(citations))
    if verified_sources:
        base += 0.07 * min(1.0, verified_sources / len(citations))
    return _clamp(base)


def model_confidence(*, intent_confidence: float, entity_confidence: float, quality_passed: bool = True) -> float:
    """How settled the generation path itself was.

    A quality-gate intervention halves this: the gate firing means the first
    answer was rejected, which is exactly the situation where the model's own
    certainty should not be taken at face value.
    """
    blended = 0.65 * intent_confidence + 0.35 * entity_confidence
    return _clamp(blended if quality_passed else blended * 0.5)


def combine(
    *,
    retrieval: float,
    source_grounding: float,
    model: float,
    reason: str = "",
) -> ConfidenceBreakdown:
    """Blend the three dimensions, with grounding as a hard ceiling."""
    blended = 0.45 * retrieval + 0.35 * source_grounding + 0.20 * model
    overall = _clamp(min(blended, source_grounding))
    if not reason:
        reason = _explain(retrieval, source_grounding, model, overall)
    return ConfidenceBreakdown(
        retrieval=_clamp(retrieval),
        source_grounding=_clamp(source_grounding),
        model=_clamp(model),
        overall=overall,
        reason=reason,
    )


def _explain(retrieval: float, grounding: float, model: float, overall: float) -> str:
    """Names the dimension that actually limited the result.

    A reason that says "confidence is low" tells a reader nothing they cannot
    see from the number. Naming the binding constraint tells them whether to
    rephrase the question, upload a document, or distrust the answer.
    """
    if grounding <= 0.0:
        return (
            "No confidence: the answer is not supported by identifiable retrieved sources, "
            "so it cannot be verified against the knowledge base."
        )
    # Name the weakest dimension, not the one that happens to be checked first.
    # A reader whose answer is well-grounded but came from an ambiguous question
    # needs to hear about the ambiguity; telling them everything is fine because
    # the blended number cleared the floor hides the one thing they could act on.
    weakest = min(
        (grounding, "grounding"),
        (retrieval, "retrieval"),
        (model, "model"),
        key=lambda pair: pair[0],
    )
    value, dimension = weakest
    if value >= HIGH_CONFIDENCE_FLOOR and overall >= HIGH_CONFIDENCE_FLOOR:
        return "Supported by well-matched, identifiable sources from the knowledge base."
    if dimension == "grounding":
        return (
            "Limited by source grounding: the retrieved sources only partly identify the law "
            "they state, so the answer cannot be traced to a specific provision with confidence."
        )
    if dimension == "retrieval":
        return (
            "Limited by retrieval: the knowledge base returned only weakly matching material "
            "for this question."
        )
    return "Limited by classification certainty: the question's intent or entities were ambiguous."
