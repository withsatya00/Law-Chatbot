"""`settings.use_neural_reranker` is off by default -- these lock in both
halves of that contract: flag off means `LegalReranker` behaves exactly as
before (no `NeuralReranker` call at all), and flag on blends the signal in
as an additive adjustment, never a replacement of the tuned heuristics, and
never breaks retrieval when the model is unavailable.
"""
import asyncio
from unittest.mock import patch

from app.core.config import settings
from app.rag.reranker import LegalReranker
from app.schemas.common import RetrievedChunk


class _FakeNeural:
    def __init__(self, scores: list[float] | None) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, texts: list[str]) -> list[float] | None:
        self.calls.append((query, list(texts)))
        return self._scores


def _chunk(text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=text, text=text, score=score, metadata={})


def test_neural_signal_is_never_consulted_when_the_flag_is_off() -> None:
    fake = _FakeNeural(scores=[1.0])
    reranker = LegalReranker(neural=fake)
    chunks = [_chunk("some legal text", 0.5)]

    with patch.object(settings, "use_neural_reranker", False):
        asyncio.run(reranker.rerank("a query", chunks, top_k=1))

    assert fake.calls == []


def test_neural_signal_blends_additively_when_enabled() -> None:
    fake = _FakeNeural(scores=[1.0])
    reranker = LegalReranker(neural=fake)
    chunk = _chunk("some legal text", 0.5)
    # No bonuses match and "a query" shares no terms with "some legal text",
    # so the heuristic-only score is just 0.75 * fused_score + 0.25 * overlap(=0).
    heuristic_only_score = 0.75 * 0.5

    with patch.object(settings, "use_neural_reranker", True), patch.object(settings, "neural_reranker_weight", 0.25):
        result = asyncio.run(reranker.rerank("a query", [chunk], top_k=1))

    assert fake.calls  # the neural signal WAS consulted this time
    # Blended, not replaced: (1 - weight) * heuristic + weight * neural,
    # strictly above the heuristic-only value for this perfect neural score.
    expected = 0.75 * heuristic_only_score + 0.25 * 1.0
    assert result[0].score == expected
    assert result[0].score > heuristic_only_score


def test_neural_signal_unavailable_leaves_heuristic_score_untouched() -> None:
    fake = _FakeNeural(scores=None)  # model unavailable / load failed
    reranker = LegalReranker(neural=fake)
    chunk = _chunk("some legal text", 0.5)

    with patch.object(settings, "use_neural_reranker", True):
        result = asyncio.run(reranker.rerank("a query", [chunk], top_k=1))

    assert fake.calls
    assert result[0].score == 0.75 * 0.5  # unchanged: heuristic score, no overlap bonus
