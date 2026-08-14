import os

from dotenv import load_dotenv
from redis import Redis


load_dotenv()


redis_client = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    decode_responses=True,#一个字节转换函数，将Redis返回的字符串解码为Python字符串
)

print(
    "Redis config:",
    redis_client.connection_pool.connection_kwargs
)