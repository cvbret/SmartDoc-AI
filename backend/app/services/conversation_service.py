import json#导入json模块，用于序列化和反序列化Python对象

from app.core.redis import redis_client


SESSION_EXPIRE_SECONDS = 3600


def _get_session_key(session_id: str) -> str:
    return f"chat:{session_id}"


def get_history(session_id: str) -> list[dict[str, str]]:
    key = _get_session_key(session_id)
    #从Redis中获取会话的所有消息
    #lrange命令用于获取列表的指定范围内的元素
    #0表示从列表的开头开始获取，-1表示获取到列表的末尾
    #ensure_ascii=False表示不将非ASCII字符转换为转义序列
    messages = redis_client.lrange(
        key,
        0,
        -1
    )

    return [
        json.loads(message)
        for message in messages
    ]


def add_message(
    session_id: str,
    role: str,
    content: str
) -> None:
    key = _get_session_key(session_id)

    message = {
        "role": role,
        "content": content
    }
    #将消息添加到Redis列表的末尾
    #rpush命令用于将元素添加到列表的末尾
    #json.dumps()将Python对象转换为JSON字符串的函数
    #ensure_ascii=False表示不将非ASCII字符转换为转义序列
    redis_client.rpush(
        key,
        json.dumps(
            message,
            ensure_ascii=False
        )
    )
    #设置会话过期时间为1小时
    #expire命令用于设置键的过期时间
    #SESSION_EXPIRE_SECONDS表示过期时间为SESSION_EXPIRE_SECONDS小时
    redis_client.expire(
        key,
        SESSION_EXPIRE_SECONDS
    )