import os
import json
import asyncio
from redis.asyncio import Redis


async def run_consumer():
    r = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    try:
        while True:
            item = await r.blpop("tasks", timeout=5)
            if item:
                _, data = item
                payload = json.loads(data)
                # Placeholder: integrate transformer model here
                print("[predictor] received task:", payload)
            else:
                await asyncio.sleep(0.1)
    finally:
        await r.close()


if __name__ == "__main__":
    asyncio.run(run_consumer())
