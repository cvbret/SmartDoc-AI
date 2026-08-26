import logging

from redis import Redis

from app.core.config import settings


logger = logging.getLogger(__name__)


redis_client = Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True,#一个字节转换函数，将Redis返回的字符串解码为Python字符串
)

logger.debug(
    "Redis client configured host=%s port=%s db=%s",
    settings.redis_host,
    settings.redis_port,
    settings.redis_db,
)
