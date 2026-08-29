from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def chat_api(app):
    """Return the chat router module after the isolated app import."""
    from app.api import chat

    return chat


def test_chat_returns_answer_and_preserves_conversation_order(
    client,
    chat_api,
    monkeypatch,
):
    session_id = "test-session"
    message = "SmartDoc AI 是什么？"
    history = [
        {
            "role": "user",
            "content": "之前的问题",
        }
    ]
    events = []

    def add_message(session, role, content):
        events.append(("add", session, role, content))

    def get_history(session):
        events.append(("history", session))
        return history

    def rag_chat(question, received_history):
        events.append(("rag", question, received_history))
        return "这是 SmartDoc AI。"

    add_message_mock = Mock(side_effect=add_message)
    get_history_mock = Mock(side_effect=get_history)
    rag_chat_mock = Mock(side_effect=rag_chat)
    monkeypatch.setattr(chat_api, "add_message", add_message_mock)
    monkeypatch.setattr(chat_api, "get_history", get_history_mock)
    monkeypatch.setattr(chat_api, "rag_chat", rag_chat_mock)

    response = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": message},
    )

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "message": "success",
        "data": {"answer": "这是 SmartDoc AI。"},
    }
    assert [event[0] for event in events] == [
        "add",
        "history",
        "rag",
        "add",
    ]
    assert events[0] == ("add", session_id, "user", message)
    assert events[1] == ("history", session_id)
    assert events[2] == ("rag", message, history)
    assert events[3] == (
        "add",
        session_id,
        "assistant",
        "这是 SmartDoc AI。",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "test-session"},
        {"message": "missing session"},
        {"session_id": "test-session", "message": None},
        {"session_id": "test-session", "message": ""},
        {"session_id": "", "message": "hello"},
    ],
)
def test_chat_rejects_invalid_request_without_calling_services(
    client,
    chat_api,
    monkeypatch,
    payload,
):
    add_message = Mock()
    get_history = Mock()
    rag_chat = Mock()
    monkeypatch.setattr(chat_api, "add_message", add_message)
    monkeypatch.setattr(chat_api, "get_history", get_history)
    monkeypatch.setattr(chat_api, "rag_chat", rag_chat)

    response = client.post("/api/chat", json=payload)

    assert response.status_code == 422
    add_message.assert_not_called()
    get_history.assert_not_called()
    rag_chat.assert_not_called()


def test_chat_returns_unified_500_when_rag_fails(
    app,
    chat_api,
    monkeypatch,
):
    add_message = Mock()
    get_history = Mock(return_value=[])
    rag_chat = Mock(side_effect=RuntimeError("test LLM failure"))
    monkeypatch.setattr(chat_api, "add_message", add_message)
    monkeypatch.setattr(chat_api, "get_history", get_history)
    monkeypatch.setattr(chat_api, "rag_chat", rag_chat)

    with TestClient(app, raise_server_exceptions=False) as error_client:
        response = error_client.post(
            "/api/chat",
            json={
                "session_id": "test-session",
                "message": "safe test question",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "code": 500,
        "message": "服务器内部错误",
        "data": None,
    }
    assert "traceback" not in response.text.lower()
    assert "test LLM failure" not in response.text
