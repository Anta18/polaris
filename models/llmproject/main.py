from fastapi import FastAPI
import uvicorn
from news_fetcher import fetch_articles, update_article
import torch
import torch.nn.functional as F
from embedder import get_embedding
from comparator import top_similar_articles, find_unique_chunks
from summarizer import summarize_unique_chunks
from fastapi.responses import JSONResponse

app = FastAPI(title="Omitted Facts Detection API", version="1.1")




@app.post("/detect_omitted_facts")
def detect_omitted_facts(data: dict):
    topic = data.get("topic", "").strip()
    if not topic:
        return {"error": "No topic provided."}

    # Step 1: Fetch articles safely
    articles = fetch_articles(topic)
    if not articles:
        return {"error": "No articles found for the given topic."}

    # Filter out any articles missing content
    articles = [a for a in articles if a.get("content")]
    if not articles:
        return {"error": "Fetched articles have no readable content."}

    contents = [a["content"] for a in articles]

    # Step 2: Generate embeddings and similarity mapping
    try:
        embeddings = get_embedding(contents)
        similar_map = top_similar_articles(embeddings, k=3)
    except Exception as e:
        return {"error": f"Embedding or similarity computation failed: {str(e)}"}

    # Step 3: Compare chunks and summarize missing (omitted) facts
    results = []
    all_omitted_segments = []

    for i, article in enumerate(articles):
        # Filter invalid neighbor indices
        neighbor_indices = [j for j in similar_map.get(i, []) if j < len(articles)]
        similar_texts = [
            articles[j]["content"] for j in neighbor_indices if articles[j].get("content")
        ]

        
        if not similar_texts:
            results.append({
                "title": article.get("title", "Untitled"),
                "url": article.get("url", ""),
                "omitted_facts": "Not enough related articles for comparison."
            })
            continue

        
        try:
            unique_segments = find_unique_chunks(article["content"], similar_texts)
        except Exception as e:
            unique_segments = []
            print(f"Chunk comparison failed for {article.get('title', 'unknown')}: {e}")
        # unique_segments is a list of dicts: {chunk, max_similarity, nearest_matches}
        all_omitted_segments.extend(unique_segments)

        
        article_max_sim = 0.0
        try:
            if neighbor_indices:
                neighbor_embs = embeddings[neighbor_indices]
                a_emb = embeddings[i].unsqueeze(0)
                a_emb_n = F.normalize(a_emb, p=2, dim=1)
                n_embs_n = F.normalize(neighbor_embs, p=2, dim=1)
                sims = torch.matmul(a_emb_n, n_embs_n.T)
                article_max_sim = float(torch.max(sims).item())
        except Exception:
            article_max_sim = 0.0

      
        chunk_texts = [seg["chunk"] for seg in unique_segments]
        summary = summarize_unique_chunks(chunk_texts)

        
        entry = {
            "title": article.get("title", "Untitled"),
            "url": article.get("url", ""),
            "omitted_facts": summary,
            "omitted_segments": unique_segments,
            "max_similarity": article_max_sim
        }

        results.append(entry)

        
        try:
            if article.get("_id"):
                updates = {
                    "omitted_chunks": chunk_texts,
                    "max_similarity": article_max_sim,
                    "omitted_summary": summary
                }
                res = update_article(article.get("_id"), updates)
                if isinstance(res, dict):
                    if res.get("ack"):
                        print(f"Updated article {article.get('_id')} in DB. modified_count={res.get('modified_count')}")

                        if res.get("doc"):
                            print(f"DB document after update: _id={res['doc'].get('_id')} title={res['doc'].get('title')}")
                    else:
                        print(f"Warning: failed to update article {article.get('_id')} in DB. response={res}")
                else:

                    if not res:
                        print(f"Warning: failed to update article {article.get('_id')} in DB")
        except Exception as e:
            print(f"DB update error for article {article.get('_id')}: {e}")

   
    combined_chunk_texts = [seg["chunk"] for seg in all_omitted_segments]
    combined_summary = summarize_unique_chunks(combined_chunk_texts)

    payload = {
        "topic": topic,
        "articles": results,
        "combined_summary": combined_summary

    }
    return JSONResponse(content=payload, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004, reload=True)

#uvicorn main:app --host 0.0.0.0 --port 8004 --reload