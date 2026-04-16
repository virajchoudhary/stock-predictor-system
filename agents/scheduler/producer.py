import os
import json
import asyncio
from redis.asyncio import Redis

redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

async def enqueue_task(payload: dict):
    await redis.rpush("tasks", json.dumps(payload))


if __name__ == "__main__":
    async def main():
        await enqueue_task({"symbol": "AAPL", "metadata": {"source": "cli"}})
        await redis.close()

    asyncio.run(main())
