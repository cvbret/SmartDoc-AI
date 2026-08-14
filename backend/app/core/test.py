from app.core.redis import redis_client


server_info = redis_client.info("server")

print("Redis run_id:", server_info.get("run_id"))
print("Redis version:", server_info.get("redis_version"))
print("Redis dbsize:", redis_client.dbsize())