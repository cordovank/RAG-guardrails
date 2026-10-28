# Phase 1

## app/ingestion/cleaner.py

Later you can chain:
boilerplate removal
profanity filters
language detection

## app/storage/

Later you can add:
QdrantStore
WeaviateStore
PGVectorStore
Without touching ingestion.

## Later...

can add:
PDFLoader
GitHubLoader
S3Loader

agent.call_tool("ingest", source="wikivoyage")
