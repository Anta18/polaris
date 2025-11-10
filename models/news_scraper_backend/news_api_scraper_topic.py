import os
import requests
import datetime
import uuid
import certifi
from pymongo import MongoClient

# -------------------- Config (env-first) --------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://siddhantsingh15032005_db_user:NL8faLkskcXUa6Nb@polaris.jkw8jwe.mongodb.net/")  # e.g. mongodb+srv://USER:PASS@cluster/db?retryWrites=true&w=majority
DB_NAME = os.getenv("DB_NAME", "polaris_news_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "articles")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "74060afc388e4b7fbf5e0f99f1ab03c9")  # rotate & set in env
NEWS_BASE_URL = os.getenv("NEWS_BASE_URL", "https://newsapi.org/v2/everything")  # <-- NO query params here

# Optional analytics endpoints...
BIAS_API_URL = os.getenv("BIAS_API_URL")
SENTIMENT_API_URL = os.getenv("SENTIMENT_API_URL")
CLICKBAIT_API_URL = os.getenv("CLICKBAIT_API_URL")
FACTCHECK_API_URL = os.getenv("FACTCHECK_API_URL")
RELIABILITY_API_URL = os.getenv("RELIABILITY_API_URL")
TOPIC_API_URL = os.getenv("TOPIC_API_URL")
SUMMARY_API_URL = os.getenv("SUMMARY_API_URL")

# -------------------- Mongo (TLS for Atlas) --------------------
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)
client.admin.command("ping")
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
collection.create_index("url", unique=True)  # de-dupe guard

# -------------------- Helpers --------------------
def _parse_published_at(s: str) -> datetime.datetime:
    if not s:
        return datetime.datetime.utcnow()
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        try:
            return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return datetime.datetime.utcnow()

def _call_api(url: str, payload: dict, timeout=15):
    try:
        if not url:
            return None
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def _run_analytics(title: str, description: str, content: str, url: str, source: str):
    """
    Calls optional analytics services. Any missing URL is skipped and defaults are used.
    """
    text_blob = " ".join(x for x in [title or "", description or "", content or ""] if x).strip()

    # Defaults match your schema (keep 'muti_source_summary' spelling to match backend)
    out = {
        "bias_classification_label": None,
        "bias_classification_probs": {},
        "bias_explain": [],

        "sentiment_analysis_label": None,
        "sentiment_analysis_probs": {},

        "clickbait_label": None,
        "clickbait_score": None,
        "clickbait_explanation": None,

        "topic": None,
        "omitted_facts_articles": [],

        "fake_news_label": None,
        "fake_news_probs": {},

        "source_reliability": None,

        "muti_source_summary": None,
        "single_source_summary": None,
    }

    base_payload = {
        "title": title,
        "description": description,
        "content": content,
        "text": text_blob,
        "url": url,
        "source": source,
    }

    # Bias
    bias = _call_api(BIAS_API_URL, base_payload)
    if bias:
        out["bias_classification_label"] = bias.get("bias_classification_label")
        out["bias_classification_probs"] = bias.get("bias_classification_probs", {}) or {}
        out["bias_explain"] = bias.get("bias_explain", []) or []

    # Sentiment
    sent = _call_api(SENTIMENT_API_URL, base_payload)
    if sent:
        out["sentiment_analysis_label"] = sent.get("sentiment_analysis_label")
        out["sentiment_analysis_probs"] = sent.get("sentiment_analysis_probs", {}) or {}

    # Clickbait
    click = _call_api(CLICKBAIT_API_URL, base_payload)
    if click:
        out["clickbait_label"] = click.get("clickbait_label")
        out["clickbait_score"] = click.get("clickbait_score")
        out["clickbait_explanation"] = click.get("clickbait_explanation")

    # Topic
    topic = _call_api(TOPIC_API_URL, base_payload)
    if topic:
        out["topic"] = topic.get("topic")

    # Fact-check / omitted facts / fake news
    fact = _call_api(FACTCHECK_API_URL, base_payload)
    if fact:
        out["fake_news_label"] = fact.get("fake_news_label")
        out["fake_news_probs"] = fact.get("fake_news_probs", {}) or {}
        out["omitted_facts_articles"] = fact.get("omitted_facts_articles", []) or []

    # Source reliability
    rel = _call_api(RELIABILITY_API_URL, base_payload)
    if rel is not None:
        # allow both direct number or {source_reliability: <num>}
        if isinstance(rel, dict):
            out["source_reliability"] = rel.get("source_reliability")
        elif isinstance(rel, (int, float)):
            out["source_reliability"] = rel

    # Summaries
    summ = _call_api(SUMMARY_API_URL, base_payload)
    if summ:
        # keep exact key spelling to match your schema
        out["muti_source_summary"] = summ.get("muti_source_summary")
        out["single_source_summary"] = summ.get("single_source_summary")

    return out

# -------------------- News fetch & save --------------------
def fetch_news(topic="gaza", from_date="2025-11-01", page_size=16):
    """Fetch news articles from NewsAPI (Everything endpoint)."""
    params = {
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "q": topic,
        "from": from_date,           # ISO date or datetime
        "sortBy": "publishedAt",
        "pageSize": page_size
    }
    resp = requests.get(NEWS_BASE_URL, params=params, timeout=20)
    if resp.status_code != 200:
        try:
            print("⚠️ Failed to fetch news:", resp.json().get("message", "Unknown error"))
        except Exception:
            print("⚠️ Failed to fetch news: Unknown error")
        return []
    return resp.json().get("articles", []) or []

def save_articles(articles):
    if not articles:
        print("⚠️ No articles to process.")
        return

    inserted = 0
    for art in articles:
        url = art.get("url")
        if not url:
            continue

        title = art.get("title") or "Untitled News"
        description = art.get("description") or "No description available."
        content = art.get("content") or "No content available."
        author = art.get("author") or "Unknown Author"
        source = (art.get("source") or {}).get("name") or "Unknown Source"
        image_url = art.get("urlToImage") or "https://via.placeholder.com/150"
        published_at = _parse_published_at(art.get("publishedAt"))

        analytics = _run_analytics(title, description, content, url, source)

        doc = {
            "id": str(uuid.uuid4()),
            "title": title,
            "description": description,
            "content": content,
            "author": author,
            "source": source,
            "url": url,
            "imageUrl": image_url,
            "publishedAt": published_at,
            "category": "General",
            "topics": [],
            "likes": 0,
            "comments": [],
            **analytics,  # bias/sentiment/clickbait/topic/omitted_facts/fake_news/source_reliability/summaries
        }

        # Upsert to avoid races/dupes
        res = collection.update_one({"url": url}, {"$setOnInsert": doc}, upsert=True)
        if res.upserted_id is not None:
            inserted += 1

    print(f"✅ {inserted} new articles stored in MongoDB")

# -------------------- Main --------------------
if __name__ == "__main__":
    arts = fetch_news(topic="gaza", from_date="2025-11-01", page_size=16)
    save_articles(arts)