"""Pure boundaries between Chroma retrieval, final selection and context."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class RetrievalCandidate:
    id: str
    text: str
    metadata: dict[str, object] | None
    # Raw Chroma distance, not a reranker relevance score.
    retrieval_distance: float
    # Raw query-passage relevance logit; higher is more relevant, not a probability.
    reranker_score: float | None = None


def _single_batch(result: Mapping[str, object], field: str) -> list:
    value = result.get(field)
    if value is None or value == []:
        return []
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise ValueError(f"Chroma {field} must contain exactly one query batch")
    return value[0]


def normalize_candidates(result: Mapping[str, object]) -> list[RetrievalCandidate]:
    """Require aligned fields; absent fields are allowed only for empty results.

    Individual metadata entries may be None (records without metadata).
    Nonempty results require IDs, text and raw distances for every record.
    """
    fields = ("ids", "documents", "metadatas", "distances")
    batches = {field: _single_batch(result, field) for field in fields}
    counts = {field: len(batch) for field, batch in batches.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Chroma candidate field counts must align: {counts}")

    candidates = []
    for index in range(counts["documents"]):
        record_id = batches["ids"][index]
        text = batches["documents"][index]
        metadata = batches["metadatas"][index]
        distance = batches["distances"][index]
        if not isinstance(record_id, str) or not isinstance(text, str):
            raise ValueError(f"Chroma candidate {index}: id and document must be strings")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"Chroma candidate {index}: metadata must be a dict or None")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise ValueError(f"Chroma candidate {index}: distance must be numeric")
        candidates.append(RetrievalCandidate(
            id=record_id,
            text=text,
            metadata=metadata.copy() if metadata is not None else None,
            retrieval_distance=float(distance),
        ))
    return candidates


def select_candidates(
    candidates: list[RetrievalCandidate], top_n: int,
) -> list[RetrievalCandidate]:
    """Preserve input order; slicing naturally caps N at candidate count."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")
    return candidates[:top_n]


def build_context(candidates: list[RetrievalCandidate]) -> str:
    return "\n".join(candidate.text for candidate in candidates)
