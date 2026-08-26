import json
from unittest.mock import Mock


def test_get_history_reads_and_decodes_messages(monkeypatch):
    from app.services import conversation_service

    redis_client = Mock()
    redis_client.lrange.return_value = [
        json.dumps(
            {"role": "user", "content": "第一条"},
            ensure_ascii=False,
        ),
        json.dumps(
            {"role": "assistant", "content": "第二条"},
            ensure_ascii=False,
        ),
    ]
    monkeypatch.setattr(
        conversation_service,
        "redis_client",
        redis_client,
    )

    result = conversation_service.get_history("test-session")

    assert result == [
        {"role": "user", "content": "第一条"},
        {"role": "assistant", "content": "第二条"},
    ]
    redis_client.lrange.assert_called_once_with(
        "chat:test-session",
        0,
        -1,
    )


def test_add_message_writes_message_and_expiration(monkeypatch):
    from app.core.config import settings
    from app.services import conversation_service

    redis_client = Mock()
    monkeypatch.setattr(
        conversation_service,
        "redis_client",
        redis_client,
    )

    conversation_service.add_message(
        "test-session",
        "assistant",
        "测试回答",
    )

    redis_client.rpush.assert_called_once_with(
        "chat:test-session",
        json.dumps(
            {"role": "assistant", "content": "测试回答"},
            ensure_ascii=False,
        ),
    )
    redis_client.expire.assert_called_once_with(
        "chat:test-session",
        settings.session_expire_seconds,
    )
