from unittest.mock import Mock

import pytest


def test_rag_chat_embeds_searches_builds_context_and_calls_llm(monkeypatch):
    from app.services import rag_service

    monkeypatch.setattr(rag_service.settings, "retrieval_k", 3)
    monkeypatch.setattr(rag_service.settings, "rerank_top_n", 3)
    question = "公司有哪些福利？"
    history = [
        {"role": "user", "content": "之前的问题"},
    ]
    query_embedding = [0.1, 0.2, 0.3]
    search_result = {
        "ids": [["id1", "id2"]],
        "documents": [["带薪年假", "五险一金"]],
        "metadatas": [[{}, {}]],
        "distances": [[0.1, 0.2]],
    }
    generate_embedding = Mock(return_value=query_embedding)
    search_chunks = Mock(return_value=search_result)
    chat_with_llm = Mock(return_value="公司提供带薪年假和五险一金。")
    monkeypatch.setattr(
        rag_service,
        "generate_embedding",
        generate_embedding,
    )
    monkeypatch.setattr(rag_service, "search_chunks", search_chunks)
    monkeypatch.setattr(rag_service, "chat_with_llm", chat_with_llm)

    answer = rag_service.rag_chat(question, history)

    assert answer == "公司提供带薪年假和五险一金。"
    generate_embedding.assert_called_once_with(question)
    search_chunks.assert_called_once_with(query_embedding, top_k=3)
    chat_with_llm.assert_called_once()
    llm_messages = chat_with_llm.call_args.args[0]
    assert llm_messages[0] == history[0]
    assert llm_messages[-1]["role"] == "user"
    assert "带薪年假\n五险一金" in llm_messages[-1]["content"]
    assert question in llm_messages[-1]["content"]


@pytest.mark.parametrize(
    "search_result",
    [
        {},
        {"documents": []},
        {"documents": [[]]},
        {"documents": None},
    ],
)
def test_rag_chat_calls_llm_with_empty_context_when_no_documents(
    monkeypatch,
    search_result,
):
    from app.services import rag_service

    question = "没有匹配资料的问题"
    generate_embedding = Mock(return_value=[0.1, 0.2, 0.3])
    search_chunks = Mock(return_value=search_result)
    chat_with_llm = Mock(return_value="不知道")
    monkeypatch.setattr(
        rag_service,
        "generate_embedding",
        generate_embedding,
    )
    monkeypatch.setattr(rag_service, "search_chunks", search_chunks)
    monkeypatch.setattr(rag_service, "chat_with_llm", chat_with_llm)

    answer = rag_service.rag_chat(question, [])

    assert answer == "不知道"
    chat_with_llm.assert_called_once()
    prompt = chat_with_llm.call_args.args[0][-1]["content"]
    assert "资料:\n\n\n\n问题:" in prompt
    assert question in prompt
    assert "带薪年假" not in prompt


def test_rag_chat_propagates_search_failure(monkeypatch):
    from app.services import rag_service

    search_chunks = Mock(side_effect=RuntimeError("test search failure"))
    monkeypatch.setattr(
        rag_service,
        "generate_embedding",
        Mock(return_value=[0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(rag_service, "search_chunks", search_chunks)
    chat_with_llm = Mock()
    monkeypatch.setattr(rag_service, "chat_with_llm", chat_with_llm)

    with pytest.raises(RuntimeError, match="test search failure"):
        rag_service.rag_chat("safe test question", [])

    chat_with_llm.assert_not_called()


def test_rag_chat_propagates_llm_failure(monkeypatch):
    from app.services import rag_service

    chat_with_llm = Mock(side_effect=RuntimeError("test LLM failure"))
    monkeypatch.setattr(
        rag_service,
        "generate_embedding",
        Mock(return_value=[0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(
        rag_service,
        "search_chunks",
        Mock(return_value={"ids": [["id1"]], "documents": [["safe test context"]],
                           "metadatas": [[None]], "distances": [[0.1]]}),
    )
    monkeypatch.setattr(rag_service, "chat_with_llm", chat_with_llm)

    with pytest.raises(RuntimeError, match="test LLM failure"):
        rag_service.rag_chat("safe test question", [])

    chat_with_llm.assert_called_once()


@pytest.fixture(autouse=True)
def fake_reranker_model(monkeypatch):
    from app.services import reranker_service
    model = Mock()
    model.predict.side_effect = lambda pairs, **kwargs: list(range(len(pairs), 0, -1))
    monkeypatch.setattr(reranker_service, "get_reranker_model", Mock(return_value=model))
    return model


def test_rag_reranks_before_final_selection(monkeypatch, fake_reranker_model):
    from app.services import rag_service

    monkeypatch.setattr(rag_service.settings, "rerank_top_n", 2)
    monkeypatch.setattr(rag_service, "generate_embedding", Mock(return_value=[0.1]))
    monkeypatch.setattr(rag_service, "search_chunks", Mock(return_value={
        "ids": [["A", "B", "C", "D"]],
        "documents": [["passage A", "passage B", "passage C", "passage D"]],
        "metadatas": [[{}, {}, {}, {}]], "distances": [[0.1, 0.2, 0.3, 0.4]],
    }))
    fake_reranker_model.predict.side_effect = None
    fake_reranker_model.predict.return_value = [0.1, 8.0, 2.0, -1.0]
    llm = Mock(return_value="answer")
    monkeypatch.setattr(rag_service, "chat_with_llm", llm)
    assert rag_service.rag_chat("question", []) == "answer"
    prompt = llm.call_args.args[0][-1]["content"]
    assert "资料:\npassage B\npassage C\n\n\n问题:" in prompt
    assert "passage A" not in prompt and "passage D" not in prompt


def test_rag_reranker_failure_does_not_call_llm(monkeypatch, fake_reranker_model):
    from app.services import rag_service
    from app.services.reranker_service import RerankerError

    monkeypatch.setattr(rag_service, "generate_embedding", Mock(return_value=[0.1]))
    monkeypatch.setattr(rag_service, "search_chunks", Mock(return_value={
        "ids": [["A"]], "documents": [["passage"]],
        "metadatas": [[None]], "distances": [[0.1]],
    }))
    fake_reranker_model.predict.side_effect = RuntimeError("inference failed")
    llm = Mock()
    monkeypatch.setattr(rag_service, "chat_with_llm", llm)
    with pytest.raises(RerankerError, match="inference"):
        rag_service.rag_chat("question", [])
    llm.assert_not_called()
