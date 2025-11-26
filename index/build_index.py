# Heading-aware chunking, embeddings, FAISS.
# This file will build and persist indexes under index/faiss_store

import argparse
import os, re, json, sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---- Config ----
JSONL_PATH = "index/data/wikivoyage_nynj_data.jsonl"     # <-- JSONL with fields: id, title, url, text
OUT_DIR    = "index/faiss_store"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_WORDS  = 500                      # ~“token-ish” budget per chunk (words)
OVERLAP    = 50                       # words of overlap between adjacent chunks (helps recall)
USE_HNSW   = False                    # set True when the corpus grows large
BATCH_SIZE = 32                       # embedding batch size (GPU memory)
# ----------------

# Match either Markdown headings (# H1) OR wiki headings (== Heading ==)
HEADING_RE = re.compile(r"^(?:#{1,6}\s+.+|=+\s*[^=\n]+?\s*=+)\s*$", re.M)

def heading_aware_chunks(text: str, max_words=MAX_WORDS, overlap=OVERLAP):
    """
    Split by headings; then pack paragraphs into ~max_words words with optional overlap.
    Works well for HTML converted to text, or Markdown.
    """
    # Split by headings if present
    sections = HEADING_RE.split(text) if HEADING_RE.search(text) else [text]
    chunks = []
    for sec in sections:
        # Paragraphs (blank-line separated)
        paras = [p.strip() for p in re.split(r"\n\s*\n", sec) if p.strip()]
        buf_words, count = [], 0
        for p in paras:
            words = p.split()
            if count + len(words) > max_words and buf_words:
                chunks.append(" ".join(buf_words))
                # Add overlap tail of previous chunk as the start of next
                tail = buf_words[-overlap:] if overlap else []
                buf_words, count = tail[:], len(tail)
            buf_words.extend(words)
            count += len(words)
        if buf_words:
            chunks.append(" ".join(buf_words))
    # Remove empties
    return [c for c in chunks if c.strip()]

def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_chunks(docs):
    meta, texts = [], []
    for d in docs:
        base_id = d.get("id")
        title   = d.get("title")
        url     = d.get("url", "")
        text    = d.get("text", "")

        for i, chunk in enumerate(heading_aware_chunks(text)):
            cid = f"{base_id}#p{i}"
            meta.append({"id": cid, "title": title, "url": url, "text": chunk})
            texts.append(chunk)
    return meta, texts

def generate_embeddings(texts, model_name, batch_size):
    model = SentenceTransformer(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # enables cosine via inner product
    ).astype("float32")
    return vecs
    
def init_faiss_index(vecs):
    """ 
    Initialize a FAISS index (HNSW or Flat).
    - FlatIP: Simple indexing. Fine for small corpora.
    - HNSW: Better for larger corpora but needs more memory.
    """
    if USE_HNSW:
        try:
            index = faiss.IndexHNSWFlat(vecs.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            # fallback to FlatIP if older FAISS versions don't support HNSW with inner-product
            index = faiss.IndexFlatIP(vecs.shape[1])
            print("HNSW with inner-product not available in this FAISS build; using IndexFlatIP.")
    else:
        index = faiss.IndexFlatIP(vecs.shape[1])
    return index

def persist_artifacts(index, vecs, meta, out_dir=OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))
    np.save(os.path.join(OUT_DIR, "vecs.npy"), vecs)  # optional
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Build FAISS index from JSONL document using --jsonl_path. If no path is provided, defaults will be used.")
    ap.add_argument("--jsonl_path", default=JSONL_PATH, required=False,
                    help=f"Path to the JSONL file (e.g., '{JSONL_PATH}').")
    args = ap.parse_args()


    """ Load docs from JSONL, chunk, embed """
    docs = list(stream_jsonl(args.jsonl_path))
    if not docs:
        print(f"No documents found in {args.jsonl_path}", file=sys.stderr)
        sys.exit(1)

    meta, texts = build_chunks(docs)
    vecs = generate_embeddings(texts, model_name=MODEL_NAME, batch_size=BATCH_SIZE)

    """ Initialize and populate FAISS index """
    index = init_faiss_index(vecs)
    index.add(vecs)

    """ Persist artifacts """
    persist_artifacts(index, vecs, meta)
    
    print(f"Built FAISS with {len(meta)} chunks from {len(docs)} docs "
          f"({len(set(m['title'] for m in meta))} unique titles)")


    """ Add to temp files cleanup """
    with open("./temp_files.txt", "a") as tf:
        tf.write(f"{os.path.join(OUT_DIR, 'faiss.index')}\n")
        tf.write(f"{os.path.join(OUT_DIR, 'vecs.npy')}\n")
        tf.write(f"{os.path.join(OUT_DIR, 'meta.json')}\n")

if __name__ == "__main__":
    main()
