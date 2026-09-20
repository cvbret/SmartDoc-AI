from copy import deepcopy
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services.retrieval_service import (
    RetrievalCandidate, normalize_candidates, select_candidates, build_context,
)


def result_for(count):
    return {
        "ids": [[f"id{i}" for i in range(count)]],
        "documents": [[f"text{i}" for i in range(count)]],
        "metadatas": [[{"document_id": i, "filename": f"file{i}", "chunk_id": i}
                       for i in range(count)]],
        "distances": [[i / 10 for i in range(count)]],
    }


def test_normalize_candidates_preserves_alignment_and_input():
    result = result_for(3)
    original = deepcopy(result)
    candidates = normalize_candidates(result)
    for i, candidate in enumerate(candidates):
        assert candidate == RetrievalCandidate(
            f"id{i}", f"text{i}", original["metadatas"][0][i], i / 10,
        )
    assert result == original


@pytest.mark.parametrize("field", ["ids", "documents", "metadatas", "distances"])
@pytest.mark.parametrize("damage", ["short", "missing", "none", "multiple_queries", "flat"])
def test_normalize_candidates_rejects_malformed_fields(field, damage):
    result = result_for(3)
    if damage == "short":
        result[field][0].pop()
    elif damage == "missing":
        del result[field]
    elif damage == "none":
        result[field] = None
    elif damage == "multiple_queries":
        result[field].append([])
    else:
        result[field] = result[field][0]
    with pytest.raises(ValueError, match="Chroma"):
        normalize_candidates(result)


@pytest.mark.parametrize("result", [{}, {"documents": []}, {"documents": [[]]},
                                   {"documents": None}, result_for(0)])
def test_normalize_candidates_empty(result):
    assert normalize_candidates(result) == []


def test_normalize_candidates_accepts_null_metadata():
    result = result_for(1)
    result["metadatas"] = [[None]]
    assert normalize_candidates(result)[0].metadata is None


@pytest.mark.parametrize("field,value", [("ids", None), ("documents", None),
                                         ("metadatas", "bad"), ("distances", None)])
def test_normalize_candidates_rejects_invalid_entries(field, value):
    result = result_for(1)
    result[field][0][0] = value
    with pytest.raises(ValueError, match="candidate 0"):
        normalize_candidates(result)


@pytest.mark.parametrize("count,top_n", [(0, 3), (1, 3), (4, 2), (3, 3), (2, 5)])
def test_select_candidates_preserves_order_without_mutating(count, top_n):
    candidates = normalize_candidates(result_for(count))
    original = candidates.copy()
    selected = select_candidates(candidates, top_n)
    assert selected == original[:top_n]
    assert selected is not candidates
    assert candidates == original


@pytest.mark.parametrize("top_n", [0, -1])
def test_select_candidates_rejects_nonpositive_n(top_n):
    with pytest.raises(ValueError, match="top_n"):
        select_candidates([], top_n)


def test_build_context_uses_only_selected_text_in_order():
    candidates = normalize_candidates(result_for(3))
    assert build_context(select_candidates(candidates, 2)) == "text0\ntext1"
    assert build_context([]) == ""


@pytest.mark.parametrize("field", ["retrieval_k", "rerank_top_n"])
@pytest.mark.parametrize("value", [0, -1])
def test_settings_rejects_nonpositive_limits(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="test", deepseek_api_key="test", **{field: value})


def test_settings_defaults_and_allows_n_greater_than_k():
    defaults = Settings(_env_file=None, database_url="test", deepseek_api_key="test",
                        retrieval_k=10, rerank_top_n=3)
    assert Settings.model_fields["retrieval_k"].default == 10
    assert Settings.model_fields["rerank_top_n"].default == 3
    assert defaults.retrieval_k == 10
    assert defaults.rerank_top_n == 3
    assert defaults.reranker_model == "BAAI/bge-reranker-base"
    config = Settings(_env_file=None, database_url="test", deepseek_api_key="test",
                      retrieval_k=2, rerank_top_n=5)
    assert len(select_candidates(normalize_candidates(result_for(2)), config.rerank_top_n)) == 2


@pytest.mark.parametrize("top_n,expected", [(2, "text0\ntext1"), (8, "text0\ntext1\ntext2")])
def test_rag_propagates_k_to_chroma_and_selects_context(monkeypatch, top_n, expected):
    from app.services import rag_service, vector_service

    collection = Mock()
    collection.query.return_value = result_for(3)
    monkeypatch.setattr(vector_service, "collection", collection)
    monkeypatch.setattr(rag_service.settings, "retrieval_k", Settings.model_fields["retrieval_k"].default)
    monkeypatch.setattr(rag_service.settings, "rerank_top_n", top_n)
    monkeypatch.setattr(rag_service, "generate_embedding", Mock(return_value=[0.1]))
    llm = Mock(return_value="answer")
    monkeypatch.setattr(rag_service, "chat_with_llm", llm)
    history = [{"role": "assistant", "content": "previous"}]
    original = deepcopy(history)
    assert rag_service.rag_chat("question", history) == "answer"
    collection.query.assert_called_once_with(
        query_embeddings=[[0.1]], n_results=10,
        include=["documents", "metadatas", "distances"],
    )
    assert llm.call_args.args[0][-1]["content"] == (
        "\n你是一个企业知识助手。\n\n请根据下面提供的资料回答问题。\n\n"
        "如果资料中没有答案，请明确说明不知道。\n\n资料:\n"
        f"{expected}\n\n\n问题:\nquestion\n"
    )
    assert history == original


@pytest.mark.parametrize("result", [{}, {"documents": []}, {"documents": [[]]}, {"documents": None}])
def test_search_chunks_logs_empty_results_safely(monkeypatch, result):
    from app.services import vector_service
    collection = Mock()
    collection.query.return_value = result
    monkeypatch.setattr(vector_service, "collection", collection)
    assert vector_service.search_chunks([0.1]) == result


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_chunks_rejects_nonpositive_k(monkeypatch, top_k):
    from app.services import vector_service
    collection = Mock()
    monkeypatch.setattr(vector_service, "collection", collection)
    with pytest.raises(ValueError, match="top_k"):
        vector_service.search_chunks([0.1], top_k=top_k)
    collection.query.assert_not_called()


def test_rag_malformed_retrieval_does_not_call_llm(monkeypatch):
    from app.services import rag_service
    monkeypatch.setattr(rag_service, "generate_embedding", Mock(return_value=[0.1]))
    monkeypatch.setattr(rag_service, "search_chunks", Mock(return_value={"documents": [["orphan"]]}))
    llm = Mock()
    monkeypatch.setattr(rag_service, "chat_with_llm", llm)
    with pytest.raises(ValueError, match="counts must align"):
        rag_service.rag_chat("question", [])
    llm.assert_not_called()


@pytest.fixture(autouse=True)
def fake_reranker_model(monkeypatch):
    from app.services import reranker_service
    model = Mock()
    model.predict.side_effect = lambda pairs, **kwargs: list(range(len(pairs), 0, -1))
    monkeypatch.setattr(reranker_service, "get_reranker_model", Mock(return_value=model))
    return model
