import json, argparse
from math import log2
from statistics import mean

from fastapi_app import retrieval as Retrieval


# Retrieval helpers

def bm25_only(query, k=5, pool=50):
    Retrieval._load_artifacts()
    idxs, _ = Retrieval._topN_bm25(query, pool)
    return [Retrieval._meta[i] for i in idxs[:k]]

def faiss_only(query, k=5, pool=50):
    Retrieval._load_artifacts()
    idxs, _, _ = Retrieval._topN_faiss(query, pool)
    return [Retrieval._meta[i] for i in idxs[:k]]

def hybrid(query, mode="hybrid", k=5, pool=50):
   return Retrieval.hybrid_search(query, k, pool)


# Metrics helpers

def is_match(passage: dict, hints: list[str]):
    """ Check if the hint is in the passage's URL or title (case-insensitive). """
    text =  (passage.get("url", "") + (passage.get("title", ""))).lower()
    return any(h in text for h in hints)

def rank_of_first_match(passages: list[dict], hints: list[str]):
    """ Return the rank (1-based) of the first passage that matches the hint.
        If no match is found, return None.
    """
    for i, p in enumerate(passages, start=1):
        if is_match(p, hints): return i
    return None

def hits_at_k(rank, k):
    """ Return 1 if rank is within top k, else 0. """
    return 1.0 if (rank is not None and rank <= k) else 0.0

def mrr(rank):
    """ Mean Reciprocal Rank: 1/rank if rank is not None, else 0. """
    return 1.0 / rank if rank is not None else 0.0

def ndcg_at_k(rank, k):
    """ Normalized Discounted Cumulative Gain at k. """
    return 1.0 / (log2(rank + 1)) if (rank is not None) and (rank <= k) else 0.0


# Main evaluation function

def evaluate(qa_file, mode="hybrid", k=5, pool=50):
    """ Evaluate retrieval using Hits@k metric based on a seed QA set.
        The QA JSONL should have at least the following fields:
        - question: The query to be evaluated.
        - doc_contains: A substring of the expected document's URL or title for matching.
    """
    retrieval_fn = {"bm25": bm25_only, "faiss": faiss_only, "hybrid": hybrid}[mode]
    R, H, M, N = [], [], [], []

    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            # Load each QA pair and retrieve documents
            qa = json.loads(line)
            query = qa.get("question")
            raw_hint = qa.get("doc_contains", "")
            hints = [h.strip().lower() for h in raw_hint if h.strip()]

            passages = retrieval_fn(query, k=k, pool=pool)

            # Compute metrics
            rank = rank_of_first_match(passages, hints)
            hits_ = hits_at_k(rank, k)
            mrr_ = mrr(rank)
            ndcg_ = ndcg_at_k(rank, k)

            # Store metrics
            R.append(rank)
            H.append(hits_)
            M.append(mrr_)
            N.append(ndcg_)

    # Aggregate results
    results = {
        "mode": mode,
        "k": k,
        "pool": pool,

        "queries": len(R),
        "hits@k": mean(H) if H else 0.0,
        "mrr": mean(M) if M else 0.0,
        "ndcg@k": mean(N) if N else 0.0,
    }
    
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate retrieval performance using Hits@k, MRR, and NDCG@k metrics. " \
    "If no arguments are provided, defaults will be used: k=5, pool=50, mode='hybrid'.")
    ap.add_argument("--qa", required=True, default="eval/fixtures/wikivoyage_nynj_qa.jsonl", type=str,
                    help="Path to the JSONL file containing seed QA pairs.")
    ap.add_argument("--k", default=5, type=int,
                    help="Number of top documents to retrieve for each question.")
    ap.add_argument("--pool", default=50, type=int,
                    help="Pool size for retrieval before applying MMR.")
    ap.add_argument("--mode", choices=["bm25", "faiss", "hybrid"], default="hybrid",
                    help="Retrieval mode: 'bm25', 'faiss', or 'hybrid'.")
    args = ap.parse_args()

    out = evaluate(args.qa, mode=args.mode, k=args.k, pool=args.pool)
    print(json.dumps(out, indent=2))

    """ Save results to a file """
    out_file = f"./eval/runs/retrieval_{args.mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


    """ Add to temp files cleanup """
    with open("./temp_files.txt", "a") as tf:
        tf.write(out_file + "\n")