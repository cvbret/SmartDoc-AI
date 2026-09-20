"""BGE query-passage scoring, separate from retrieval and final selection."""

from dataclasses import replace
from functools import lru_cache
from threading import Lock

import numpy as np

from app.core.config import settings
from app.services.retrieval_service import RetrievalCandidate


class RerankerError(RuntimeError):
    """Reranking failed; callers must not silently use retrieval order."""


_model_lock = Lock()


@lru_cache(maxsize=1)
def _load_reranker_model():
    try:
        from sentence_transformers import CrossEncoder
        from torch.nn import Identity

        return CrossEncoder(
            settings.reranker_model, activation_fn=Identity(), max_length=512,
        )
    except Exception as exc:
        raise RerankerError(f"Failed to load reranker {settings.reranker_model}") from exc


def get_reranker_model():
    # Serialize the first cache miss too: lru_cache alone can load twice concurrently.
    with _model_lock:
        return _load_reranker_model()


def rerank_candidates(
    query: str, candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    if not candidates:
        return []

    model = get_reranker_model()
    pairs = [(query, candidate.text) for candidate in candidates]
    try:
        raw_scores = model.predict(
            pairs, convert_to_numpy=True, show_progress_bar=False, apply_softmax=False,
        )
    except Exception as exc:
        raise RerankerError("Reranker inference failed") from exc

    try:
        scores = np.asarray(raw_scores)
        if scores.shape != (len(candidates),):
            raise RerankerError(
                f"Reranker score shape {scores.shape}; expected ({len(candidates)},)"
            )
        if scores.dtype.kind not in "fiu":
            raise RerankerError("Reranker scores must be real numbers")
        scores = scores.astype(float)
        if not np.isfinite(scores).all():
            raise RerankerError("Reranker scores must be finite")
    except (TypeError, ValueError, OverflowError) as exc:
        raise RerankerError("Invalid reranker scores") from exc

    scored = [replace(candidate, reranker_score=float(scores[index]))
              for index, candidate in enumerate(candidates)]
    return sorted(scored, key=lambda candidate: candidate.reranker_score, reverse=True)
