import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis.asyncio import Redis

app = FastAPI(title="Scheduler API")

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis = Redis.from_url(redis_url)


class EnqueueRequest(BaseModel):
    symbol: str
    metadata: dict = {}


@app.on_event("shutdown")
async def shutdown():
    await redis.close()


@app.post("/enqueue/")
async def enqueue(req: EnqueueRequest):
    payload = {"symbol": req.symbol, "metadata": req.metadata}
    try:
        await redis.rpush("tasks", json.dumps(payload))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "enqueued", "payload": payload}
