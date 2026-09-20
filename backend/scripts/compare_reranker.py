r"""Small manual comparison, not a benchmark or answer evaluation.

From backend: ..\venv\Scripts\python.exe -B -m scripts.compare_reranker
Optional: --output PATH writes the complete evidence only when requested.
Model imports are lazy so comparison arithmetic can be tested without models.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from app.services.retrieval_service import RetrievalCandidate, normalize_candidates


CASES_PATH = Path(__file__).with_name("reranker_comparison_cases.json")


def load_cases(path: Path = CASES_PATH) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    documents, queries = cases["documents"], cases["queries"]
    ids = [document["id"] for document in documents]
    query_ids = [query["id"] for query in queries]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Corpus must have nonempty, unique document IDs")
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("Cases must have nonempty, unique query IDs")
    for query in queries:
        if query["expected_relevant_id"] not in ids:
            raise ValueError(f"{query['id']}: expected ID is absent from corpus")
    return cases


def find_rank(candidates: list[RetrievalCandidate], target_id: str) -> int | None:
    """One-based rank by record ID; None means not retrieved."""
    return next((i for i, candidate in enumerate(candidates, 1)
                 if candidate.id == target_id), None)


def compare_rankings(
    baseline: list[RetrievalCandidate], reranked: list[RetrievalCandidate],
    expected_id: str, top_n: int,
) -> dict:
    """Compare the same pool; positive baseline-reranked delta is improvement.

    Each demo query has exactly one manually chosen primary relevant ID.
    A missing ID is a candidate retrieval boundary, not a reranker failure.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    baseline_ids = [candidate.id for candidate in baseline]
    reranked_ids = [candidate.id for candidate in reranked]
    if len(set(baseline_ids)) != len(baseline_ids) or len(set(reranked_ids)) != len(reranked_ids):
        raise ValueError("Candidate IDs must be unique in each ranking")
    if set(baseline_ids) != set(reranked_ids):
        raise ValueError("Rankings must contain exactly the same candidate IDs")
    baseline_rank = find_rank(baseline, expected_id)
    reranked_rank = find_rank(reranked, expected_id)
    delta = None if baseline_rank is None else baseline_rank - reranked_rank
    outcome = ("NOT_RETRIEVED" if delta is None else
               "IMPROVED" if delta > 0 else "DEGRADED" if delta < 0 else "UNCHANGED")
    return {
        "baseline_rank": baseline_rank,
        "reranked_rank": reranked_rank,
        "rank_delta": delta,
        "baseline_top_n_hit": baseline_rank is not None and baseline_rank <= top_n,
        "reranked_top_n_hit": reranked_rank is not None and reranked_rank <= top_n,
        "outcome": outcome,
    }


def summarize(results: list[dict]) -> dict:
    outcomes = Counter(result["outcome"] for result in results)
    return {
        "cases": len(results),
        **{name: outcomes[name] for name in ("IMPROVED", "UNCHANGED", "DEGRADED", "NOT_RETRIEVED")},
        "baseline_top_n_hits": sum(result["baseline_top_n_hit"] for result in results),
        "reranked_top_n_hits": sum(result["reranked_top_n_hit"] for result in results),
    }


def format_report(report: dict) -> str:
    lines = ["Reranker comparison (small demo evidence)",
             "Rank delta = baseline rank - reranked rank; positive means improvement.",
             json.dumps(report["environment"], ensure_ascii=False, indent=2)]
    top_n = report["environment"]["rerank_top_n"]
    for result in report["results"]:
        lines += ["=" * 60, f"{result['id']}: {result['question']}",
                  f"Expected: {result['expected_relevant_id']}"]
        for label in ("baseline", "reranked"):
            lines.append(f"{label.title()} (full candidate ranking; * = Top-{top_n}):")
            for rank, candidate in enumerate(result[label], 1):
                marker = "*" if rank <= top_n else " "
                lines.append(
                    f"{marker}{rank}. {candidate['id']} | distance={candidate['retrieval_distance']:.6f}"
                    f" | reranker_score={candidate['reranker_score']} | {candidate['text']}"
                )
        delta = result["rank_delta"]
        delta_text = "N/A" if delta is None else f"{delta:+d}"
        lines += [f"Expected rank: {result['baseline_rank']} -> {result['reranked_rank']}; delta={delta_text}",
                  f"Top-{top_n} hit: {result['baseline_top_n_hit']} -> {result['reranked_top_n_hit']}",
                  f"Outcome: {result['outcome']}"]
        if result["outcome"] == "NOT_RETRIEVED":
            lines.append("Candidate Retrieval recall problem: reranking cannot recover a missing chunk.")
    lines += ["Summary:", json.dumps(report["summary"], ensure_ascii=False, indent=2)]
    return "\n".join(lines)


def run_comparison() -> dict:
    cases = load_cases()
    started = time.perf_counter()
    print("Loading real embedding and reranker models...", file=sys.stderr, flush=True)
    import chromadb
    from app.core.config import settings
    from app.services import embedding_service, reranker_service

    reranker = reranker_service.get_reranker_model()
    # Never import vector_service: its module creates a persistent client.
    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"reranker-comparison-{uuid4().hex}", embedding_function=None)
    try:
        documents = cases["documents"]
        collection.add(
            ids=[document["id"] for document in documents],
            documents=[document["text"] for document in documents],
            metadatas=[{"topic": document["topic"]} for document in documents],
            embeddings=[embedding_service.generate_embedding(document["text"]) for document in documents],
        )
        results = []
        for query in cases["queries"]:
            print(f"Comparing {query['id']}...", file=sys.stderr, flush=True)
            raw = collection.query(
                query_embeddings=[embedding_service.generate_embedding(query["question"])],
                n_results=settings.retrieval_k,
                include=["documents", "metadatas", "distances"],
            )
            baseline = normalize_candidates(raw)
            # Snapshot before scoring, preserving exactly Chroma's original order.
            baseline_rows = [asdict(candidate) for candidate in baseline]
            reranked = reranker_service.rerank_candidates(query["question"], baseline)
            results.append({
                **query,
                **compare_rankings(baseline, reranked, query["expected_relevant_id"], settings.rerank_top_n),
                "baseline": baseline_rows,
                "reranked": [asdict(candidate) for candidate in reranked],
            })
        return {
            "environment": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "embedding_device": str(embedding_service.model.device),
                "reranker_device": str(reranker.model.device),
                "chroma_client": "EphemeralClient (unique collection, default L2 distance)",
                "versions": {name: version(name) for name in ("chromadb", "sentence-transformers", "torch", "transformers")},
                "chunk_count": len(documents), "query_count": len(results),
                "retrieval_k": settings.retrieval_k, "rerank_top_n": settings.rerank_top_n,
                "corpus_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            "results": results, "summary": summarize(results),
        }
    finally:
        client.delete_collection(collection.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Explicit JSON evidence destination; no file is written by default")
    args = parser.parse_args()
    report = run_comparison()
    print(format_report(report))
    if args.output is not None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
