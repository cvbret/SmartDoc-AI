from redis import Redis

from app.core.config import settings


redis_client = Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True,#一个字节转换函数，将Redis返回的字符串解码为Python字符串
)

print(
    "Redis config:",
    redis_client.connection_pool.connection_kwargs
)
