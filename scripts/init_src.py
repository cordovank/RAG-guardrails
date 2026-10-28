from pathlib import Path

SRC = Path("src")

paths = [
    # app
    SRC / "app",
    SRC / "app/api",
    SRC / "app/core",
    SRC / "app/rag",
    SRC / "app/storage",
    SRC / "app/ingestion",
    SRC / "app/prompts",

    # eval
    SRC / "eval/datasets",

    # misc
    SRC / "experiments",
    SRC / "scripts",
    SRC / "tests/unit",
    SRC / "tests/integration",
]

files = [
    # app
    SRC / "app/__init__.py",
    SRC / "app/main.py",

    # api
    SRC / "app/api/__init__.py",
    SRC / "app/api/health.py",
    SRC / "app/api/query.py",
    SRC / "app/api/ingest.py",

    # core
    SRC / "app/core/__init__.py",
    SRC / "app/core/config.py",
    SRC / "app/core/logging.py",
    SRC / "app/core/tracing.py",
    SRC / "app/core/metrics.py",

    # rag
    SRC / "app/rag/__init__.py",
    SRC / "app/rag/pipeline.py",
    SRC / "app/rag/retriever.py",
    SRC / "app/rag/embedder.py",
    SRC / "app/rag/reranker.py",
    SRC / "app/rag/generator.py",
    SRC / "app/rag/schemas.py",

    # storage
    SRC / "app/storage/__init__.py",
    SRC / "app/storage/faiss_store.py",
    SRC / "app/storage/bm25_store.py",
    SRC / "app/storage/document_store.py",

    # ingestion
    SRC / "app/ingestion/__init__.py",
    SRC / "app/ingestion/loader.py",
    SRC / "app/ingestion/chunker.py",
    SRC / "app/ingestion/cleaner.py",
    SRC / "app/ingestion/indexer.py",

    # prompts
    SRC / "app/prompts/__init__.py",
    SRC / "app/prompts/base_prompt.txt",
    SRC / "app/prompts/citation_prompt.txt",

    # eval
    SRC / "eval/__init__.py",
    SRC / "eval/metrics.py",
    SRC / "eval/run_eval.py",
    SRC / "eval/baselines.py",
    SRC / "eval/datasets/eval_set.json",

    # scripts
    SRC / "scripts/ingest_docs.py",
    SRC / "scripts/rebuild_index.py",
    SRC / "scripts/export_embeddings.py",

    # tests
    SRC / "tests/unit/test_chunker.py",
    SRC / "tests/unit/test_retriever.py",
    SRC / "tests/unit/test_pipeline.py",
    SRC / "tests/integration/test_query_api.py",
]

for p in paths:
    p.mkdir(parents=True, exist_ok=True)

for f in files:
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch(exist_ok=True)

print("✅ src/ skeleton created successfully")
