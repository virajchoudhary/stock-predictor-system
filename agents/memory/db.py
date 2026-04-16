import os
from motor.motor_asyncio import AsyncIOMotorClient

_mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/stock_memory")
_client = AsyncIOMotorClient(_mongo_uri)
_db = _client.get_default_database() or _client["stock_memory"]


async def save_prediction(doc: dict):
    """Insert a prediction document into the `predictions` collection."""
    return await _db.predictions.insert_one(doc)
