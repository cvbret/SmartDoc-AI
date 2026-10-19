"""Sequential offline execution; no production vector-store imports."""

from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import logging
import os
import platform

from app.services.prompt_service import build_rag_messages
from app.services.retrieval_service import (
    normalize_candidates, select_candidates, build_context,
)
from evaluation.dataset import load_evaluation_dataset
from evaluation.index import evaluation_index, query_index
from evaluation.metrics import METRIC_NAMES, create_metrics, safe_error, score_metrics


@contextmanager
def private_provider_logs():
    """Offline runs report safe structured errors instead of raw SDK tracebacks.

    This scope is process-wide: run the CLI in its own process, never inside an
    API request. Restore the previous logging level even after setup failures.
    """
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def choose_samples(samples: list[dict], limit: int | None, sample_ids: list[str] | None) -> list[dict]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if sample_ids is not None:
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample IDs must be nonempty and unique")
        if set(sample_ids) - {sample["id"] for sample in samples}:
            raise ValueError("Unknown sample ID")
        samples = [sample for sample in samples if sample["id"] in sample_ids]
    return samples[:] if limit is None else samples[:limit]


async def evaluate_sample(
    sample: dict, contexts: dict[str, str], collection, *, mode: str,
    retrieval_k: int, top_n: int, embed: Callable, answer: Callable,
    rerank: Callable, metrics: dict,
) -> dict:
    row = {
        "sample_id": sample["id"], "question": sample["question"],
        "reference_answer": sample["reference_answer"],
        "reference_context_ids": sample["reference_context_ids"].copy(),
        "reference_contexts": [contexts[ref] for ref in sample["reference_context_ids"]],
        "configured_retrieval_k": retrieval_k, "actual_candidate_count": None,
        "candidate_context_ids": [], "retrieved_context_ids": [], "retrieved_contexts": [],
        "actual_answer": None, "metrics": dict.fromkeys(METRIC_NAMES),
        "metric_status": dict.fromkeys(METRIC_NAMES, "skipped"), "errors": [],
        "status": "failed",
    }
    try:
        candidates = normalize_candidates(query_index(collection, embed(sample["question"]), retrieval_k))
        for candidate in candidates:
            if candidate.id not in contexts or candidate.text != contexts[candidate.id]:
                raise ValueError("Retrieved ID/text does not match the fixed corpus")
        row["actual_candidate_count"] = len(candidates)
        row["candidate_context_ids"] = [candidate.id for candidate in candidates]
    except Exception as exc:
        row["errors"].append(safe_error("retrieval", exc))
        return row
    if mode == "reranker":
        try:
            candidates = rerank(sample["question"], candidates)
        except Exception as exc:
            row["errors"].append(safe_error("reranking", exc))
            return row
    selected = select_candidates(candidates, top_n)
    row["retrieved_context_ids"] = [candidate.id for candidate in selected]
    row["retrieved_contexts"] = [candidate.text for candidate in selected]
    try:
        actual = answer(build_rag_messages(sample["question"], [], build_context(selected)))
        if not isinstance(actual, str) or not actual.strip():
            raise ValueError("Answer generation returned empty or invalid content")
        row["actual_answer"] = actual
    except Exception as exc:
        row["errors"].append(safe_error("answer_generation", exc))
    # Context metrics have no dependency on actual_answer and can still succeed.
    await score_metrics(row, metrics)
    row["status"] = "completed" if not row["errors"] else "partial"
    return row


def summarize_results(rows: list[dict], configured_count: int) -> dict:
    summary = {}
    for name in METRIC_NAMES:
        values = [row["metrics"][name] for row in rows if row["metric_status"][name] == "success"]
        summary[name] = {
            "configured_count": configured_count, "successful_count": len(values),
            "failed_count": sum(row["metric_status"][name] == "failed" for row in rows),
            "skipped_count": sum(row["metric_status"][name] == "skipped" for row in rows),
            "mean_of_successful_values": sum(values) / len(values) if values else None,
        }
    return summary


async def run_evaluation(
    mode: str = "reranker", limit: int | None = None, sample_ids: list[str] | None = None,
) -> dict:
    if mode not in ("baseline", "reranker"):
        raise ValueError("mode must be baseline or reranker")
    data, contexts = load_evaluation_dataset()
    samples = choose_samples(data["samples"], limit, sample_ids)
    # Runtime libraries are lazy: --help and deterministic tests load no models.
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    with private_provider_logs():
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from app.core.config import settings
        from app.services.embedding_service import generate_embedding, model as retrieval_model
        from app.services.llm_service import chat_with_llm
        from app.services.reranker_service import rerank_candidates

        async with AsyncOpenAI(
            api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
            timeout=60, max_retries=0,
        ) as client:
            judge = llm_factory(
                "deepseek-chat", provider="openai", client=client, adapter="instructor",
                temperature=0, max_tokens=2048, max_retries=2,
            )
            embeddings = HuggingFaceEmbeddings(
                model="BAAI/bge-small-zh-v1.5", device="cpu", normalize_embeddings=True,
            )
            metrics = create_metrics(judge, embeddings)
            rows = []
            with evaluation_index(contexts, generate_embedding) as collection:
                for sample in samples:
                    rows.append(await evaluate_sample(
                        sample, contexts, collection, mode=mode,
                        retrieval_k=settings.retrieval_k, top_n=settings.rerank_top_n,
                        embed=generate_embedding, answer=chat_with_llm,
                        rerank=rerank_candidates, metrics=metrics,
                    ))
                    print(f"{sample['id']}: {rows[-1]['status']}", flush=True)
            metadata = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_mode": mode,
                "dataset_version": data["dataset_version"], "corpus_version": data["corpus_version"],
                "corpus_sha256": data["corpus_sha256"], "chunking": data["chunking"],
                "retrieval_k": settings.retrieval_k, "configured_retrieval_k": settings.retrieval_k,
                "rerank_top_n": settings.rerank_top_n, "corpus_context_count": len(contexts),
                "retrieval_embedding_model": settings.embedding_model,
                "retrieval_device": str(retrieval_model.device),
                "reranker_model": settings.reranker_model if mode == "reranker" else None,
                "answer_model": "deepseek-chat", "judge_model": "deepseek-chat",
                "evaluation_embedding_model": "BAAI/bge-small-zh-v1.5",
                "evaluation_embedding_device": "cpu", "evaluation_embedding_normalize": True,
                "judge_temperature": 0, "judge_max_tokens": 2048, "judge_max_retries": 2,
                "judge_api_configured": bool(settings.deepseek_api_key),
                "ragas_version": version("ragas"), "python_version": platform.python_version(),
                "dependency_versions": {name: version(name) for name in (
                    "instructor", "datasets", "langchain-community", "openai", "pydantic",
                    "sentence-transformers", "transformers", "torch", "numpy",
                )},
                "dataset_sample_count": len(data["samples"]), "configured_sample_count": len(samples),
                "completed_sample_count": sum(row["status"] == "completed" for row in rows),
                "processed_sample_count": len(rows), "sample_ids": [sample["id"] for sample in samples],
                "limitations": [
                    "Same-model judging may introduce shared-model bias and common blind spots.",
                    "The small 5-chunk corpus does not represent broad retrieval performance.",
                ],
            }
    return {"metadata": metadata, "samples": rows, "summary": summarize_results(rows, len(samples))}
