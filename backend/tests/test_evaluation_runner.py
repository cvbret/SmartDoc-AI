import asyncio
from contextlib import contextmanager
from copy import deepcopy
import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.prompt_service import build_rag_messages
from evaluation import runner
from evaluation.dataset import load_evaluation_dataset
from evaluation.index import evaluation_index, query_index
from evaluation.metrics import METRIC_NAMES


def raw_result():
    return {"ids": [["A", "B", "C"]], "documents": [["text A", "text B", "text C"]],
            "metadatas": [[{}, {}, {}]], "distances": [[0.1, 0.2, 0.3]]}


@pytest.fixture
def execution(monkeypatch):
    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace(SingleTurnSample=lambda **kw: SimpleNamespace(**kw)))
    sample = {"id": "EVAL-X", "question": "question", "reference_answer": "ref B",
              "reference_context_ids": ["B"]}
    contexts = {"A": "text A", "B": "text B", "C": "text C"}
    collection = Mock()
    collection.count.return_value = 3
    collection.query.return_value = raw_result()
    deps = dict(mode="baseline", retrieval_k=10, top_n=1,
                embed=Mock(return_value=[0.1]), answer=Mock(return_value="actual"),
                rerank=Mock(side_effect=lambda q, rows: [rows[1], rows[0], rows[2]]),
                metrics={name: SimpleNamespace(
                    ascore=AsyncMock(return_value=SimpleNamespace(value=0.5)),
                    single_turn_ascore=AsyncMock(return_value=1.0),
                ) for name in METRIC_NAMES})
    return sample, contexts, collection, deps


@pytest.mark.parametrize("mode,selected", [("baseline", "A"), ("reranker", "B")])
def test_modes_use_same_pool_and_only_final_context_in_prompt_and_metrics(execution, mode, selected):
    sample, contexts, collection, deps = execution
    deps["mode"] = mode
    before = deepcopy(sample)
    row = asyncio.run(runner.evaluate_sample(sample, contexts, collection, **deps))
    assert row["candidate_context_ids"] == ["A", "B", "C"]
    assert row["actual_candidate_count"] == 3
    assert row["configured_retrieval_k"] == 10
    assert row["retrieved_context_ids"] == [selected]
    assert row["retrieved_contexts"] == [contexts[selected]]
    deps["answer"].assert_called_once_with(build_rag_messages("question", [], contexts[selected]))
    id_sample = deps["metrics"]["context_precision_id"].single_turn_ascore.await_args.args[0]
    assert id_sample.retrieved_context_ids == [selected]
    assert collection.query.call_args.kwargs["n_results"] == 3
    if mode == "baseline":
        deps["rerank"].assert_not_called()
    else:
        assert [c.id for c in deps["rerank"].call_args.args[1]] == ["A", "B", "C"]
    assert row["status"] == "completed"
    assert sample == before


@pytest.mark.parametrize("stage", ["retrieval", "reranking", "answer_generation"])
def test_stage_failures_are_isolated_and_next_sample_runs(execution, stage):
    sample, contexts, collection, deps = execution
    if stage == "retrieval":
        collection.query.side_effect = [RuntimeError("dummy-secret"), raw_result()]
    elif stage == "reranking":
        deps["mode"] = "reranker"
        deps["rerank"].side_effect = [RuntimeError("dummy-secret"), []]
    else:
        deps["answer"].side_effect = [RuntimeError("dummy-secret"), "second answer"]
    async def run():
        first = await runner.evaluate_sample(sample, contexts, collection, **deps)
        second = await runner.evaluate_sample(sample, contexts, collection, **deps)
        return first, second
    first, second = asyncio.run(run())
    assert first["errors"][0]["stage"] == stage
    assert first["actual_answer"] is None
    assert first["metrics"]["faithfulness"] is None
    assert second["actual_answer"] is not None
    assert "dummy-secret" not in json.dumps([first, second])
    if stage != "answer_generation":
        assert all(status == "skipped" for status in first["metric_status"].values())
    else:
        assert first["metric_status"]["context_recall_id"] == "success"


@pytest.mark.parametrize("damage", ["unknown_id", "wrong_text"])
def test_runtime_identity_must_match_ground_truth(execution, damage):
    sample, contexts, collection, deps = execution
    raw = raw_result()
    raw["ids" if damage == "unknown_id" else "documents"][0][0] = "wrong"
    collection.query.return_value = raw
    row = asyncio.run(runner.evaluate_sample(sample, contexts, collection, **deps))
    assert row["errors"][0]["stage"] == "retrieval"
    deps["answer"].assert_not_called()


def test_summary_counts_success_failure_skips_and_keeps_zero(execution):
    sample, contexts, collection, deps = execution
    row = asyncio.run(runner.evaluate_sample(sample, contexts, collection, **deps))
    rows = [deepcopy(row) for _ in range(3)]
    name = "faithfulness"
    rows[0]["metrics"][name] = 0.0
    rows[1]["metric_status"][name] = "failed"
    rows[1]["metrics"][name] = None
    rows[2]["metric_status"][name] = "skipped"
    rows[2]["metrics"][name] = None
    summary = runner.summarize_results(rows, 3)[name]
    assert summary == dict(configured_count=3, successful_count=1, failed_count=1,
                           skipped_count=1, mean_of_successful_values=0.0)
    assert runner.summarize_results([], 0)[name]["mean_of_successful_values"] is None


@pytest.mark.parametrize("fail", [False, True])
def test_index_uses_ephemeral_stable_ids_and_cleans_up(monkeypatch, fail):
    client, collection = Mock(), Mock()
    collection.name = "isolated"
    client.create_collection.return_value = collection
    chroma = SimpleNamespace(EphemeralClient=Mock(return_value=client), PersistentClient=Mock())
    monkeypatch.setitem(sys.modules, "chromadb", chroma)
    contexts = {"redis::chunk_000": "redis evidence"}
    embed = Mock(return_value=[0.2])
    try:
        with evaluation_index(contexts, embed) as found:
            assert found is collection
            if fail:
                raise RuntimeError("sample failed")
    except RuntimeError:
        assert fail
    chroma.EphemeralClient.assert_called_once_with()
    chroma.PersistentClient.assert_not_called()
    assert collection.add.call_args.kwargs["ids"] == list(contexts)
    assert collection.add.call_args.kwargs["documents"] == list(contexts.values())
    assert collection.add.call_args.kwargs["embeddings"] == [[0.2]]
    client.delete_collection.assert_called_once_with("isolated")


def test_index_cleans_up_if_embedding_fails(monkeypatch):
    client = Mock()
    client.create_collection.return_value.name = "isolated"
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(EphemeralClient=lambda: client))
    with pytest.raises(RuntimeError):
        with evaluation_index({"A": "text"}, Mock(side_effect=RuntimeError("failed"))):
            pytest.fail("must not yield")
    client.delete_collection.assert_called_once_with("isolated")


def test_empty_index_does_not_query_zero_results():
    collection = Mock()
    collection.count.return_value = 0
    assert query_index(collection, [0.1], 10) == {}
    collection.query.assert_not_called()
    with pytest.raises(ValueError):
        query_index(collection, [0.1], 0)


def test_shared_prompt_preserves_history_and_exact_text():
    history = [{"role": "assistant", "content": "earlier"}]
    before = deepcopy(history)
    result = build_rag_messages("Q", history, "A\nB")
    assert result is not history and history == before
    assert result == history + [{"role": "user", "content":
        "\n你是一个企业知识助手。\n\n请根据下面提供的资料回答问题。\n\n"
        "如果资料中没有答案，请明确说明不知道。\n\n资料:\nA\nB\n\n\n问题:\nQ\n"}]


def test_sample_selection_supports_all_twenty_and_targeted_smoke():
    data, _ = load_evaluation_dataset()
    assert len(runner.choose_samples(data["samples"], None, None)) == 20
    selected = runner.choose_samples(data["samples"], 2, ["EVAL-010", "EVAL-019"])
    assert [s["id"] for s in selected] == ["EVAL-010", "EVAL-019"]


@pytest.mark.parametrize("limit,ids", [(0, None), (-1, None), (None, ["unknown"]),
                                      (None, ["EVAL-001", "EVAL-001"]), (None, [])])
def test_invalid_sample_selection_fails_before_models(limit, ids):
    data, _ = load_evaluation_dataset()
    with pytest.raises(ValueError):
        runner.choose_samples(data["samples"], limit, ids)


def test_private_logs_suppress_raw_secrets_and_restore_level(caplog):
    previous = logging.root.manager.disable
    with pytest.raises(RuntimeError):
        with runner.private_provider_logs():
            logging.error("Authorization: dummy-secret")
            raise RuntimeError("failure")
    assert "dummy-secret" not in caplog.text
    assert logging.root.manager.disable == previous


def test_full_runner_metadata_and_all_samples_with_doubles(monkeypatch):
    from app.services import embedding_service, llm_service
    data, contexts = load_evaluation_dataset()
    monkeypatch.setattr(embedding_service, "model", SimpleNamespace(device="cpu"))
    monkeypatch.setattr(embedding_service, "generate_embedding", Mock(return_value=[0.1]))
    monkeypatch.setattr(llm_service, "chat_with_llm", Mock(return_value="answer"))
    class AsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    async_factory = Mock(return_value=AsyncClient())
    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", async_factory)
    factory = Mock(return_value=object())
    embedding_factory = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "ragas.llms", SimpleNamespace(llm_factory=factory))
    monkeypatch.setitem(sys.modules, "ragas.embeddings", SimpleNamespace(HuggingFaceEmbeddings=embedding_factory))
    monkeypatch.setattr(runner, "create_metrics", Mock(return_value={}))
    @contextmanager
    def index(*args): yield object()
    monkeypatch.setattr(runner, "evaluation_index", index)
    async def sample_result(sample, *args, **kwargs):
        return {"status": "completed", "metrics": dict.fromkeys(METRIC_NAMES, 0.5),
                "metric_status": dict.fromkeys(METRIC_NAMES, "success")}
    evaluate = AsyncMock(side_effect=sample_result)
    monkeypatch.setattr(runner, "evaluate_sample", evaluate)
    report = asyncio.run(runner.run_evaluation("baseline"))
    assert evaluate.await_count == 20
    meta = report["metadata"]
    assert meta["dataset_version"] == data["dataset_version"]
    assert meta["corpus_sha256"] == data["corpus_sha256"]
    assert meta["run_mode"] == "baseline"
    assert meta["answer_model"] == meta["judge_model"] == "deepseek-chat"
    assert meta["evaluation_embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert meta["retrieval_k"] > 0 and meta["rerank_top_n"] > 0
    assert meta["configured_sample_count"] == meta["completed_sample_count"] == 20
    assert meta["reranker_model"] is None
    factory.assert_called_once()
    assert factory.call_args.kwargs["adapter"] == "instructor"
    assert factory.call_args.kwargs["temperature"] == 0
    embedding_factory.assert_called_once_with(model="BAAI/bge-small-zh-v1.5", device="cpu", normalize_embeddings=True)
    json.dumps(report, allow_nan=False)
