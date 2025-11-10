Omitted Facts Detection API
===========================

Overview
--------
The Omitted Facts Detection API compares news articles about a topic, detects unique or under-reported facts, and generates comparative summaries. It runs as a FastAPI service defined in `main.py`.

Base URL
--------
Run the service with uvicorn (example command below). Default base URL: `http://127.0.0.1:8000`.

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Authentication
--------------
No authentication is enforced by default. Ensure the host is secured if exposing publicly.

Endpoint Summary
----------------
- `POST /detect_omitted_facts` – Fetches articles, computes semantic similarities, isolates potentially omitted facts, and returns article-level summaries.

Request
-------
- **Headers**: `Content-Type: application/json`
- **Body** (`application/json`):

  - `topic` *(string, required)*: Keyword or short phrase describing the news topic to analyze.

Example Request Body
--------------------
```
{
  "topic": "global semiconductor supply chain"
}
```

Response
--------
On success, the API returns a JSON object with per-article summaries of omitted facts and a combined view.

- `topic` *(string)*: Echoes the supplied topic after trimming.
- `articles` *(array)*: Each fetched article enriched with comparative details.
  - `title` *(string)*: Article title.
  - `url` *(string)*: Article URL.
  - `omitted_facts` *(string)*: Summary of facts potentially missing or under-reported elsewhere. Defaults to a fallback message if no unique evidence was detected.
  - `omitted_segments` *(array)*: Raw evidence objects for each potentially unique chunk. Each object contains `chunk`, `max_similarity`, and `nearest_matches` (with peer snippets and similarity scores).
- `combined_summary` *(string)*: Global summary across all omitted segments.

Example Success Response
------------------------
```
{
  "topic": "global semiconductor supply chain",
  "articles": [
    {
      "title": "Beijing targets consultancy ...",
      "url": "https://example.com/beijing",
      "omitted_facts": "Highlights government pressure on teardown consultancies that other coverage overlooks."
    },
    {
      "title": "Suppliers prepare for iPhone 18",
      "url": "https://example.com/iphone",
      "omitted_facts": "No significant omitted facts detected."
    }
  ],
  "combined_summary": "Export controls and reverse-engineering crackdowns emerge as the most unique risk factors."
}
```

Error Responses
---------------
- `{"error": "No topic provided."}` – Topic missing or empty.
- `{"error": "No articles found for the given topic."}` – Fetcher returned zero results.
- `{"error": "Fetched articles have no readable content."}` – All results lacked usable text.
- `{"error": "Embedding or similarity computation failed: ..."}` – Adapter or device failure during embedding.

Internal Workflow
-----------------
1. `news_fetcher.fetch_articles(topic)` pulls recent coverage.
2. `embedder.get_embedding(contents)` produces adapter-enhanced MiniLM embeddings.
3. `comparator.top_similar_articles` selects nearest neighbours for each article.
4. `comparator.find_unique_chunks` identifies article passages not mirrored elsewhere.
5. `summarizer.summarize_unique_chunks` uses FLAN-T5-base to craft comparative summaries.
6. `main.detect_omitted_facts` assembles and returns the response.

Dependencies
------------
Required packages are listed in `requirements.txt` (`fastapi`, `uvicorn`, `torch`, `transformers`, `sentence-transformers`, `nltk`, etc.). Ensure `NEWS_API_KEY` is set in `config.py` for article retrieval.

Operational Tips
----------------
- Provide descriptive topics for richer comparisons.
- Maintain `residual_adapters.pt` aligned with `MiniLMWithAdapters` to benefit from fine-tuning.
- Increase `config.NUM_ARTICLES` for broader comparisons if performance allows.
- Consider caching article results to reduce repeated external calls when experimenting.

