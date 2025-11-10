import requests
from config import NEWS_API_KEY, NUM_ARTICLES, MONGO_URI, MONGO_DB, MONGO_COLLECTION
from typing import List, Dict
import os

try:
    from pymongo import MongoClient
    from bson import ObjectId
except Exception:
    MongoClient = None
    ObjectId = None


def _query_mongo(topic: str) -> List[Dict]:
    """Return list of articles from MongoDB matching topic (case-insensitive).
    Each article dict will include a string '_id' if present in DB.
    """
    if MongoClient is None:
        return []

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    coll = db[MONGO_COLLECTION]

    cursor = coll.find({"topic": {"$regex": topic, "$options": "i"}})

    results = []
    for doc in cursor:
        results.append({
            "_id": str(doc.get("_id")),
            "title": doc.get("title", "Untitled"),
            "url": doc.get("url", ""),
            "content": doc.get("content", ""),
            "source": "mongo"
        })

    client.close()
    return results


def update_article(article_id: str, updates: Dict):
    """Update article document in MongoDB by _id (string)."""
    if MongoClient is None:
        print("pymongo not available; cannot update DB.")
        return {"ack": False, "doc": None}

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    coll = db[MONGO_COLLECTION]
    try:
        _id = ObjectId(article_id)
    except Exception:
        _id = article_id

    res = coll.update_one({"_id": _id}, {"$set": updates})

    updated = coll.find_one({"_id": _id})
    if updated:
  
        try:
            updated["_id"] = str(updated["_id"])
        except Exception:
            pass
    client.close()
    return {"ack": bool(res.acknowledged), "modified_count": getattr(res, 'modified_count', None), "doc": updated}


def fetch_articles(topic: str) -> List[Dict]:
    """Fetch articles for a topic. Try MongoDB first; fall back to NewsAPI.
    Returns a list of dicts with keys: title, url, content, optionally _id.
    """
    # First try MongoDB
    try:
        mongo_articles = _query_mongo(topic)
    except Exception as e:
        mongo_articles = []
        print(f"MongoDB query failed: {e}")

    if mongo_articles:
        return mongo_articles

    # Fallback to NewsAPI
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "pageSize": NUM_ARTICLES,
        "apiKey": NEWS_API_KEY
    }
    r = requests.get(url, params=params)
    data = r.json()
    articles = []
    for art in data.get("articles", []):
        articles.append({
            "title": art.get("title"),
            "url": art.get("url"),
            "content": art.get("description") or "",
            "source": "newsapi"
        })
    return articles
