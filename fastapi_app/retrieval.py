"""
Hybrid Retrieval: BM25 + FAISS + RRF + MMR

- FAISS: semantic similarity via vector search
- BM25: exact term matching
- RRF: Reciprocal Rank Fusion to combine BM25 + FAISS results
- MMR: Maximal Marginal Relevance to re-rank final results for diversity
- SentenceTransformer: to embed queries for FAISS

Tuning parameters:
    POOL:   number of candidates taken from each retriever before fusion.
            ↑ improves recall (more to fuse), can increase latency.
    RRF_K:  RRF constant for fusion steepness.
            ↑ improves diversity (more even scores), can decrease precision.
    MMR_LAMBDA: MMR diversity constant for re-ranking.
            ↑ improves relevance (less diverse), can decrease diversity.

Requires:
    pip install faiss-cpu rank_bm25 sentence-transformers
    (or faiss-gpu if you have a CUDA GPU)

Goal: return a list of (doc_id, score) tuples for a given query.
Only the top-k results are returned, but more candidates are pulled from each retriever
to improve recall before fusion and re-ranking.
"""

import os, json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Globals
FAISS_DIR = "index/faiss_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
POOL = 50           # pull candidates from each retriever before fusion
RRF_K = 60          # RRF constant (usually 60-100; higher = more diverse)
MMR_LAMBDA = 0.7    # MMR diversity constant (0 = diverse, 1 = relevance)

# ---- lazy singletons to avoid reloading per request ----
_faiss = None       # FAISS index
_meta = None        # list of dicts: {id,title,url,text}
_vecs = None        # doc vectors for MMR (shape [n, d])
_stmodel = None     # SentenceTransformer model
_bm25 = None        # BM25Okapi over tokenized documents


def _load_artifacts():
    """Load FAISS index + metadata + vectors + BM25 tokenizer + embedder model."""
    global _faiss, _meta, _vecs, _bm25, _stmodel
    if all(v is not None for v in [_faiss, _meta, _vecs, _bm25, _stmodel]):
        return  # already loaded

    _faiss = faiss.read_index(f"{FAISS_DIR}/faiss.index")
    _meta = json.load(open(f"{FAISS_DIR}/meta.json", "r", encoding="utf-8"))
    
    _vecs = np.load(f"{FAISS_DIR}/vecs.npy").astype("float32")
    _stmodel = SentenceTransformer(EMBED_MODEL)

    # Prepare BM25
    corpus = [m["text"].split(" ") for m in _meta]
    _bm25 = BM25Okapi(corpus=corpus)

def _topN_faiss(query: str, n: int):
    """ Retrieve top-n candidates semantically similar to query via FAISS. """
    query_vec = _stmodel.encode([query], normalize_embeddings=True).astype("float32")
    faiss_results = _faiss.search(query_vec, n)
    scores, idx = faiss_results 

    return idx[0].tolist(), scores[0].tolist(), query_vec[0]

def _topN_bm25(query: str, n: int):
    """ Retrieve top-n candidates lexically similar to query via BM25. """
    tokens = query.split()
    scores = _bm25.get_scores(tokens) 
    idx = np.argsort(scores)[::-1][:n]

    return idx.tolist(), scores[idx].tolist()

def _rrf(rank_lists, k=RRF_K):
    """ Combine FAISS + BM25 results with Reciprocal Rank Fusion (RRF). 
    
    rank_lists: list of lists of (idx, score) tuples ordered by rank
    Steps:
        1. For each list, convert ranks to RRF scores: 1 / (k + rank)
        2. Sum RRF scores across lists for each candidate
        3. Return candidates ordered by combined RRF score

    Returns: list of (idx, combined_rrf_score) tuples ordered by combined RRF score
    """
    combined_scores = {}
    for r in rank_lists:
        for rank, idx in enumerate(r):
            rrf_score = 1 / (k + rank + 1)  # rank is 0-based
            combined_scores[idx] = combined_scores.get(idx, 0.0) + rrf_score

    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

def _mmr(candidates: list, query_vec: np.ndarray, top_k: int, lam=MMR_LAMBDA):
    """ Re-rank candidates with Maximal Marginal Relevance (MMR) for diversity. 
    
    MMR balances relevance to the query and diversity among the selected documents.
    It removes near-duplicates and chooses the top-k most relevant yet diverse documents.
    
    How it works:
        - Iteratively select candidates that maximize MMR score:
           MMR = λ * Sim(query, doc) - (1 - λ) * Div(doc)
           where: Div = max(Sim(doc, selected))
        - Stop when top_k candidates are selected. 
    """

    # Fallback: if no vectors, return candidates as-is
    if _vecs is None or query_vec is None or len(candidates) == 0:
        return candidates[:top_k]

    selected, candidates = [], set(candidates)
    while len(selected) < min(top_k, len(candidates)):
        best_candidate, best_score = None, -float("inf")

        for c in candidates:
            rel = float(np.dot(query_vec, _vecs[c]))
            # _vecs[selected] -> shape (m, d); _vecs[c] #-> shape (d,)
            div = float(np.dot(_vecs[selected], _vecs[c]).max()) if selected else 0.0
            score = lam*rel - (1.0-lam)*div

            # stream best candidate
            if score > best_score:
                best_candidate, best_score = c, score
        
        selected.append(best_candidate)
        candidates.remove(best_candidate)
    
    return selected


def hybrid_search(
        query: str, 
        k: int = 5, 
        pool: int = POOL
        ) -> list:
    """
    Hybrid search orchestrator using BM25 + FAISS + RRF + MMR. 

    Steps:
    1. Embed query with SentenceTransformer.
    2. Retrieve top-POOL from FAISS (semantic) and BM25 (exact match).
    3. Combine with Reciprocal Rank Fusion (RRF).
    4. Re-rank with Maximal Marginal Relevance (MMR) for diversity.
    5. Return top-k results.

    Args:
        query: user query string
        k: number of results to return
        pool: number of candidates to pull from each retriever before fusion

    Returns: list of top-k (doc_id, score) tuples for the query
    """

    # 1. Lazy load artifacts
    _load_artifacts()

    # 2. Retrieve top-POOL from FAISS and BM25
    faiss_idx, _, query_vec = _topN_faiss(query, pool)
    bm25_idx, _ = _topN_bm25(query, pool)

    # 3. Fuse with RRF
    fused = _rrf([faiss_idx, bm25_idx], k=RRF_K)

    # 4. Re-rank with MMR
    fused_candidates = [idx for idx, _ in fused]    
    selected = _mmr(fused_candidates, query_vec, top_k=k, lam=MMR_LAMBDA)

    # 5. Return top-k results as metadata dicts
    # return [("doc1#p1", 1.0), ("doc2#p3", 0.9)][:k]   # placeholder
    return [_meta[i] for i in selected]
