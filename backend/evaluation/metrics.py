"""Modern semantic scoring and one narrowly scoped legacy ID adapter."""

import math
from numbers import Real


METRIC_NAMES = (
    "faithfulness", "answer_relevancy", "context_precision_llm",
    "context_recall_llm", "context_precision_id", "context_recall_id",
)
ANSWER_METRICS = {"faithfulness", "answer_relevancy"}


def safe_error(stage: str, exc: Exception) -> dict:
    # Never serialize provider exception text: it may contain credentials, headers,
    # prompts or entire HTTP bodies. Stage, type and HTTP status remain diagnostic.
    error = {
        "stage": stage, "error_type": type(exc).__name__,
        "message": f"{stage} failed; provider exception details omitted for credential safety.",
    }
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        error["http_status"] = status
    return error


def create_metrics(judge, embeddings) -> dict:
    from ragas.metrics.collections import (
        Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
    )
    # 0.4.3 does not export the ID metrics from collections despite its warning.
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=3),
        "context_precision_llm": ContextPrecision(llm=judge),
        "context_recall_llm": ContextRecall(llm=judge),
        "context_precision_id": IDBasedContextPrecision(),
        "context_recall_id": IDBasedContextRecall(),
    }


async def score_metric(name: str, metric, row: dict) -> float:
    if name in ("context_precision_id", "context_recall_id"):
        from ragas import SingleTurnSample

        # LLM ContextPrecision is rank-sensitive semantic AP; ID precision is a
        # set overlap proportion. Keep these distinct in results and interpretation.
        sample = SingleTurnSample(
            retrieved_context_ids=row["retrieved_context_ids"],
            reference_context_ids=row["reference_context_ids"],
        )
        value = await metric.single_turn_ascore(sample)
    else:
        inputs = {"user_input": row["question"]}
        if name in ANSWER_METRICS:
            inputs["response"] = row["actual_answer"]
        else:
            inputs["reference"] = row["reference_answer"]
        if name != "answer_relevancy":
            inputs["retrieved_contexts"] = row["retrieved_contexts"]
        value = (await metric.ascore(**inputs)).value
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Metric returned a non-finite or non-numeric value")
    return float(value)


async def score_metrics(row: dict, metrics: dict) -> None:
    for name in METRIC_NAMES:
        if row["actual_answer"] is None and name in ANSWER_METRICS:
            continue
        try:
            row["metrics"][name] = await score_metric(name, metrics[name], row)
            row["metric_status"][name] = "success"
        except Exception as exc:
            row["metrics"][name] = None
            row["metric_status"][name] = "failed"
            row["errors"].append(safe_error(name, exc))
