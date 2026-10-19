import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from evaluation.metrics import METRIC_NAMES, create_metrics, safe_error, score_metrics


@pytest.fixture
def scoring(monkeypatch):
    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace(SingleTurnSample=lambda **kw: SimpleNamespace(**kw)))
    row = {
        "question": "question", "actual_answer": "actual", "reference_answer": "reference",
        "retrieved_context_ids": ["B"], "reference_context_ids": ["B", "C"],
        "retrieved_contexts": ["selected B"], "metrics": dict.fromkeys(METRIC_NAMES),
        "metric_status": dict.fromkeys(METRIC_NAMES, "skipped"), "errors": [],
    }
    metrics = {name: SimpleNamespace(
        ascore=AsyncMock(return_value=SimpleNamespace(value=0.5)),
        single_turn_ascore=AsyncMock(return_value=0.5),
    ) for name in METRIC_NAMES}
    return row, metrics


def test_metric_inputs_match_verified_signatures_and_final_ids(scoring):
    row, metrics = scoring
    asyncio.run(score_metrics(row, metrics))
    metrics["faithfulness"].ascore.assert_awaited_once_with(
        user_input="question", response="actual", retrieved_contexts=["selected B"],
    )
    metrics["answer_relevancy"].ascore.assert_awaited_once_with(user_input="question", response="actual")
    for name in ("context_precision_llm", "context_recall_llm"):
        metrics[name].ascore.assert_awaited_once_with(
            user_input="question", reference="reference", retrieved_contexts=["selected B"],
        )
    for name in ("context_precision_id", "context_recall_id"):
        sample = metrics[name].single_turn_ascore.await_args.args[0]
        assert sample.retrieved_context_ids == ["B"]
        assert sample.reference_context_ids == ["B", "C"]


def test_metric_failure_is_null_zero_is_valid_and_others_continue(scoring):
    row, metrics = scoring
    metrics["faithfulness"].ascore.side_effect = TimeoutError("dummy-secret")
    metrics["answer_relevancy"].ascore.return_value.value = 0.0
    asyncio.run(score_metrics(row, metrics))
    assert row["metrics"]["faithfulness"] is None
    assert row["metrics"]["answer_relevancy"] == 0.0
    assert row["metric_status"]["answer_relevancy"] == "success"
    assert all(row["metric_status"][name] == "success" for name in METRIC_NAMES[1:])
    assert row["errors"][0]["stage"] == "faithfulness"
    assert "dummy-secret" not in json.dumps(row)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, "0.5", True])
@pytest.mark.parametrize("name", ["faithfulness", "context_precision_id"])
def test_invalid_metric_values_become_null_with_error(scoring, value, name):
    row, metrics = scoring
    if name.endswith("_id"):
        metrics[name].single_turn_ascore.return_value = value
    else:
        metrics[name].ascore.return_value.value = value
    asyncio.run(score_metrics(row, metrics))
    assert row["metrics"][name] is None
    assert row["metric_status"][name] == "failed"
    assert row["errors"][0]["error_type"] == "ValueError"
    json.dumps(row, allow_nan=False)


def test_missing_answer_skips_only_answer_metrics(scoring):
    row, metrics = scoring
    row["actual_answer"] = None
    asyncio.run(score_metrics(row, metrics))
    for name in METRIC_NAMES[:2]:
        metrics[name].ascore.assert_not_awaited()
        assert row["metric_status"][name] == "skipped"
    assert all(row["metric_status"][name] == "success" for name in METRIC_NAMES[2:])


def test_safe_errors_never_serialize_secrets_or_provider_bodies():
    error = RuntimeError('Authorization: Bearer dummy-api-key; response={"key":"dummy-api-key"}')
    error.status_code = 401
    result = safe_error("answer_generation", error)
    assert result["http_status"] == 401
    assert result["error_type"] == "RuntimeError"
    assert "dummy-api-key" not in json.dumps(result)
    assert "Authorization" not in json.dumps(result)


def test_metric_constructors_use_modern_classes_and_isolated_id_adapter(monkeypatch):
    modern = SimpleNamespace(**{name: Mock() for name in (
        "Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall",
    )})
    legacy = SimpleNamespace(IDBasedContextPrecision=Mock(), IDBasedContextRecall=Mock())
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", modern)
    monkeypatch.setitem(sys.modules, "ragas.metrics", legacy)
    judge, embeddings = object(), object()
    metrics = create_metrics(judge, embeddings)
    assert set(metrics) == set(METRIC_NAMES)
    modern.AnswerRelevancy.assert_called_once_with(llm=judge, embeddings=embeddings, strictness=3)
    for name in ("Faithfulness", "ContextPrecision", "ContextRecall"):
        getattr(modern, name).assert_called_once_with(llm=judge)
    legacy.IDBasedContextPrecision.assert_called_once_with()
    legacy.IDBasedContextRecall.assert_called_once_with()
