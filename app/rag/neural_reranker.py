"""Optional cross-encoder relevance signal for `LegalReranker`.

Off by default (`settings.use_neural_reranker`) -- see that setting's
docstring for why this stays an opt-in extra signal rather than replacing
any of `LegalReranker`'s hand-tuned heuristic bonuses. Mirrors
`app.rag.embeddings.EmbeddingProvider`'s lazy, class-cached model loading:
models are heavy to load, so one `CrossEncoder` per model name is shared
across every `NeuralReranker()` in the process, and inference (CPU/GPU
bound, synchronous) always runs on a worker thread.
"""
from __future__ import annotations

import asyncio
import math
import threading
from typing import TYPE_CHECKING, ClassVar

import structlog

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = structlog.get_logger(__name__)


class NeuralReranker:
    _model_cache: ClassVar[dict[str, CrossEncoder]] = {}
    _load_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.neural_reranker_model

    async def score(self, query: str, texts: list[str]) -> list[float] | None:
        """Relevance score per text, in [0, 1], or `None` if the model
        can't be used right now. Never raises -- a missing model/download
        failure/inference error degrades to "no neural signal this call"
        rather than breaking retrieval, same as the BM25/Atlas legs in
        `MongoVectorStore.search`.
        """
        if not texts:
            return []
        try:
            return await asyncio.to_thread(self._score_sync, query, texts)
        except Exception as exc:  # noqa: BLE001 - an optional signal must never break retrieval
            log.warning("neural_reranker_unavailable", model=self.model_name, error=str(exc))
            return None

    def _score_sync(self, query: str, texts: list[str]) -> list[float]:
        model = self._get_model()
        raw_scores = model.predict([(query, text) for text in texts])
        # Cross-encoder logits aren't bounded to [0, 1] the way every other
        # bonus in `LegalReranker.rerank` is; a sigmoid puts this signal on
        # the same scale before it's blended in.
        return [1.0 / (1.0 + math.exp(-float(value))) for value in raw_scores]

    def _get_model(self) -> CrossEncoder:
        cached = NeuralReranker._model_cache.get(self.model_name)
        if cached is not None:
            return cached
        with NeuralReranker._load_lock:
            cached = NeuralReranker._model_cache.get(self.model_name)
            if cached is not None:
                return cached
            from sentence_transformers import CrossEncoder

            log.info("neural_reranker_model_loading", model=self.model_name)
            model: CrossEncoder = CrossEncoder(self.model_name)
            NeuralReranker._model_cache[self.model_name] = model
            log.info("neural_reranker_model_loaded", model=self.model_name)
            return model
