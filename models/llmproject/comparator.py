import re

import torch
import torch.nn.functional as F

from embedder import get_embedding

_SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def simple_sentence_split(text: str):
    """Lightweight sentence splitter relying on punctuation boundaries."""

    if not text:
        return []

    sentences = _SENTENCE_BOUNDARY_REGEX.split(text)
    cleaned = []

    for sentence in sentences:
        stripped = sentence.strip()
        if stripped:
            cleaned.append(stripped)

    return cleaned

def chunk_text(text, max_sentences=3, overlap=1):
    """Chunk text with overlapping sentences for better context preservation"""
    if not text:
        return []
        
    try:
        sentences = simple_sentence_split(text)
            
        chunks = []
        for i in range(0, len(sentences), max_sentences - overlap):
            chunk = ' '.join(sentences[i:i + max_sentences])
            if len(chunk.strip()) > 40:  # Minimum chunk size
                chunks.append(chunk.strip())
        
        return chunks
    except Exception as e:
        print(f"Warning: Falling back to basic text chunking due to error: {str(e)}")
        
        words = text.split()
        chunks = []
        chunk_size = 100  # roughly 20 words
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 40:
                chunks.append(chunk.strip())
        return chunks

def semantic_similarity(embeddings_a, embeddings_b):
    """Calculate cosine similarity between two embeddings."""

    if not isinstance(embeddings_a, torch.Tensor):
        embeddings_a = torch.tensor(embeddings_a, dtype=torch.float32)
    if not isinstance(embeddings_b, torch.Tensor):
        embeddings_b = torch.tensor(embeddings_b, dtype=torch.float32)

    embeddings_a = F.normalize(embeddings_a.view(1, -1), dim=1)
    embeddings_b = F.normalize(embeddings_b.view(1, -1), dim=1)

    return float(torch.matmul(embeddings_a, embeddings_b.T).item())

def top_similar_articles(embeddings, k=3, threshold=0.4):
    """Find top-k similar articles using semantic similarity"""
    n = len(embeddings)
    similar = {}
    
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        emb_matrix = embeddings.clone().detach()
    else:
        emb_matrix = torch.tensor(embeddings, dtype=torch.float32)

    if emb_matrix.ndim == 1:
        emb_matrix = emb_matrix.unsqueeze(0)

    emb_matrix = F.normalize(emb_matrix, p=2, dim=1)
    similarity_matrix = torch.matmul(emb_matrix, emb_matrix.T)
    
    for i in range(n):
        scores = similarity_matrix[i]
        values, indices = torch.topk(scores, k + 1)

        neighbours = []
        for idx, score in zip(indices.tolist(), values.tolist()):
            if idx == i:
                continue
            if score >= threshold or len(neighbours) < k:
                neighbours.append(idx)
            if len(neighbours) == k:
                break

        if not neighbours:
            sorted_indices = torch.argsort(scores, descending=True)
            neighbours = [idx.item() for idx in sorted_indices if idx.item() != i][:k]

        similar[i] = neighbours
    
    return similar

def find_unique_chunks(base_text, other_texts, threshold=0.72, top_k_matches=3):
    base_chunks = chunk_text(base_text)
    all_other_chunks = [chunk for text in other_texts for chunk in chunk_text(text)]
    if not base_chunks or not all_other_chunks:
        return []

    base_embs = get_embedding(base_chunks)
    other_embs = get_embedding(all_other_chunks)

    if base_embs.numel() == 0 or other_embs.numel() == 0:
        return []

    similarity_matrix = torch.matmul(base_embs, other_embs.T)
    unique_segments = []

    for idx, chunk in enumerate(base_chunks):
        sims = similarity_matrix[idx]
        max_score, max_index = torch.max(sims, dim=0)
        max_val = float(max_score.item()) if not torch.isnan(max_score) else 0.0

        top_k = min(top_k_matches, sims.size(0))
        nearest = []
        if top_k > 0:
            top_values, top_indices = torch.topk(sims, top_k)
            for value, j in zip(top_values.tolist(), top_indices.tolist()):
                nearest.append({
                    "text": all_other_chunks[j],
                    "similarity": float(value)
                })

        if max_val < threshold:
            unique_segments.append({
                "chunk": chunk,
                "max_similarity": max_val,
                "nearest_matches": nearest
            })

    return unique_segments
