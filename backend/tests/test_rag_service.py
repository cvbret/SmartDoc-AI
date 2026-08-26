from unittest.mock import Mock

import pytest


def test_rag_chat_embeds_searches_builds_context_and_calls_llm(monkeypatch):
    from app.services import rag_service

    question = "公司有哪些福利？"
    history = [
        {"role": "user", "content": "之前的问题"},
    ]
    query_embedding = [0.1, 0.2, 0.3]
    search_result = {
        "documents": [["带薪年假", "五险一金"]],
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
    search_chunks.assert_called_once_with(query_embedding)
    chat_with_llm.assert_called_once()
    llm_messages = chat_with_llm.call_args.args[0]
    assert llm_messages[0] == history[0]
    assert llm_messages[-1]["role"] == "user"
    assert "带薪年假\n五险一金" in llm_messages[-1]["content"]
    assert question in llm_messages[-1]["content"]


def test_rag_chat_calls_llm_with_empty_context_when_no_documents(
    monkeypatch,
):
    from app.services import rag_service

    question = "没有匹配资料的问题"
    generate_embedding = Mock(return_value=[0.1, 0.2, 0.3])
    search_chunks = Mock(return_value={"documents": [[]]})
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
    assert "资料:" in prompt
    assert question in prompt
    assert "带薪年假" not in prompt


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
        Mock(return_value={"documents": [["safe test context"]]}),
    )
    monkeypatch.setattr(rag_service, "chat_with_llm", chat_with_llm)

    with pytest.raises(RuntimeError, match="test LLM failure"):
        rag_service.rag_chat("safe test question", [])

    chat_with_llm.assert_called_once()
