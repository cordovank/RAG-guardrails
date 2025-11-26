# RAG System with Guardrails
This repository contains code for building a Retrieval-Augmented Generation (RAG) system with guardrails features using FastAPI, FAISS, and guardrails. The system supports hybrid retrieval (BM25 + FAISS) with Reciprocal Rank Fusion (RRF) and Maximal Marginal Relevance (MMR), answer generation with citations, and a self-correction mechanism (WIP).

***Default topic:** Travel & Places (Wikivoyage NYC + NJ).*


## Architecture Overview

- FastAPI service exposes `POST /retrieve` and `POST /answer`, plus `/health` and `/metrics`. 
- Pydantic models in `schemas.py` define strict request/response contracts, keeping API outputs predictable.
- Config is centralized in `settings.py` via `BaseSettings`, allowing overrides through environment variables (max docs, snippet length, LLM endpoints, etc.).

## Setup & Run
We use `conda` for environment management. The `environment.yml` file specifies the dependencies. To set up the environment, the FastAPI app, and use Ollama (free local LLM) for inference, follow these steps:

**Create & activate conda environment:**
```bash
conda env create -f environment.yml
conda activate rag
```

**Run FastAPI app:**
```bash
uvicorn fastapi_app.main:app --reload
```

**Ollama setup:**
Ollama is a free alternative to Paid APIs. It runs locally. It can run open-source models, and it provides an API endpoint on your computer that is compatible with OpenAI.

First, download Ollama by visiting: https://ollama.com

Then, on a separate terminal, run:

```bash
conda activate rag
# if first time, pull a model (recommended: llama3.2)
ollama pull llama3.2
ollama run llama3.2
```

**Load data and build index:**
On a separate terminal, run:
```bash
python index/fetcher.py --predefined_domain wikivoyage_nynj
python index/build_index.py
```

Try a sample retrieval and answering request using curl on the terminal. For more details, see below for sample payloads. Note that samples may vary based on your indexed data.

```bash
curl -s -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" -d '{"q":"What are some entertainment options offered?","k":5}'

curl -s -X POST http://localhost:8000/answer -H "Content-Type: application/json" -d '{"q":"What are some entertainment options offered?", "k":5}'
```

For a quick end-to-end smoke test, run:
```bash
./scripts/smoke.sh
```

## FastAPI RAG Endpoints & Payloads

### Retrieval endpoint
`POST /retrieve`: Retrieve top-k relevant passages for a given query.

- **Retrieve request:**
    POST /retrieve
    ```json
    {
    "q": "What are some entertainment options offered?",
    "k": 5
    }
    ```

- **Retrieve response:**
    ```json
    {
    "query": "What are some entertainment options offered?",
    "passages": [
        {"id":"wikivoyage_nynj_New_York_City#p68", "title":"New_York_City","url":"https://en.wikivoyage.org/wiki/New_York_City","text":"[edit]### Entertainment [edit]#### Theater and performing arts...","score":null
        },
        // more passages...
    ],
    "meta":{"elapsed_ms":29.4285,"used_k":5}
    }
    ```

### Answering
`POST /answer`: Generate an answer with citations based on the retrieved passages.

- **Answer request:**
    POST /answer
    ```json
    {
    "q": "What are some entertainment options offered?",
    "k": 5
    }
    ```

- **Answer response:**
    ```json
    {
    "query": "What are some entertainment options offered?",
    "answer": "New York City offers an enormous number and variety of...",
    "citations": [{
      "id":"wikivoyage_nynj_New_York_City#p68", 
      "title":"New_York_City",
      "url":"https://en.wikivoyage.org/wiki/New_York_City",
      "snippet":"[edit]### Entertainment [edit]#### Theater and performing arts [edit]New York boasts an enormous number and variety of theatrical performances.",
      "score":0.0}, 
      // more citations...
      ],
    "meta":{
      "citation_map":{
        "1":"https://en.wikivoyage.org/wiki/New_York_City#p68",
        // more citation mappings...
        },
      "citation_text":"Citations: \n[1] Passage: https://en.wikivoyage.org/wiki/New_York_City#p68\n - [2]...",
      "used_citation_count":2,
      "used_citation_numbers":["1","4"],
      "used_passage_ids":[
        "https://en.wikivoyage.org/wiki/New_York_City#p68",
        // more used passage ids...
        ],
      "used_passages":5,
      "elapsed_ms":3529.8452919814736}
    }
    ```

---

# Module Walkthrough of the RAG System

## Core RAG Pipeline
- `rag_pipeline.py` orchestrates end-to-end answering: enforce governance → hybrid retrieval → LLM synthesis → annotate meta like `used_passages`.

- Governance (`governance.py`) is a lightweight regex filter for PII/toxicity, applied on both input and output; violations raise 403 in the API.

- Hybrid retrieval (`retrieval.py`) lazily loads FAISS index, SentenceTransformer, metadata, and BM25 model once per process.
    - FAISS search (semantic) + BM25 (lexical) each pull a candidate pool.
    - Reciprocal Rank Fusion combines ranks; Maximal Marginal Relevance re-ranks for diversity.
    - Final results return metadata dicts (id/title/url/text/score).

- Answer synthesis (`answerer.py`) prepares numbered snippets, crafts an instruction-heavy prompt, calls an OpenAI-compatible client (Ollama by default, custom override optional), parses numeric citations, and returns structured answer + citation metadata. There’s a hook for self-checking, for future implementation.


## Data ingestion & indexing
We pull curated Wikivoyage pages (or custom URLs), clean them, emit JSONL manifests plus seed QA pairs for evaluation. Then, chunk them with heading-aware logic, embed with SentenceTransformers, and build a FAISS index. Generated are tracked for cleanup.
Additional details below. 

### 1) Data Ingestion
Choose a compact corpus (15 html pages max) on a topic of interest. Here, we used a pre-defined topic "Travel & Places" from Wikivoyage, focused on New York & New Jersey borroughs, but other topics are possible. We provide a seed set of 15 questions for evaluation.

- **Pre-defined topics**
Run the following to fetch the data and create a seed QA set:

```bash
python index/fetcher.py --predefined_domain wikivoyage_nynj
```

This will create manifest files `index/data/wikivoyage_nynj_data.jsonl` (the cleaned text), as well as `eval/fixtures/wikivoyage_nynj_qa.jsonl` (the seed QA set).

- **Custom URLs**
If you want to try your own topic, create a text file with up to 15 URLs (one URL per line) and add a few seed questions (one question per line, with optional `;`-separated list of doc titles that should contain the answer). 

Then run the following (replace the paths as needed):

```bash
python index/fetcher.py --custom_urls index/data/my_urls.txt --custom_qa index/data/my_qa.txt
```

### 2) Chunking & Indexing

<!-- Heading-aware chunking, embeddings, FAISS. -->

Next, build the FAISS index. This will create `index/faiss_store/faiss.index`, `vecs.npy` (optional), and `meta.json` (the chunk metadata).

What it does:
- Chunking: Splits the document into smaller, semantically coherent chunks.
- Embeddings: Converts text chunks into dense vector representations using a pre-trained model.
- FAISS: Builds an efficient similarity search index over the embeddings.

Justification:
- Chunking helps in retrieving more relevant passages by breaking down large documents.
- Using a pre-trained model like `all-MiniLM-L6-v2` provides good quality embeddings for semantic search. (Alternatively, `all-mpnet-base-v2` can be used for better quality at the cost of speed.)
- FAISS is a widely used library for efficient similarity search, making it suitable for production use cases.


- Chunking: We use heading-aware chunking to create semantically coherent chunks that fit within typical LLM context windows (e.g., 500 tokens). This helps improve retrieval relevance by ensuring that the most pertinent information is included in each chunk. The chunking strategy balances chunk size and overlap to optimize for both retrieval accuracy and efficiency. (We use heading-aware chunking to preserve context around section titles, which often contain important cues about the content.)
- Embeddings: We use a pre-trained SentenceTransformer model to convert text chunks into dense vector representations. This allows us to capture semantic similarity between the query and document chunks. It is a good trade-off between performance and computational efficiency, making it suitable for real-time applications.
- FAISS: FAISS provides a scalable solution for handling large datasets, ensuring that the retrieval process remains fast even as the corpus grows. We use FAISS to build an efficient similarity search index over the embeddings. This enables fast retrieval of relevant chunks at query time.

Overall, this indexing strategy is designed to enhance the quality of retrieved passages, which is crucial for building a reliable RAG system.

```bash
python index/build_index.py
uvicorn fastapi_app.main:app --reload
curl -s -X POST "http://localhost:8000/retrieve?q=Your%20Question&k=5"
```

---

## 3) Retrieval

**Hybrid Retrieval (BM25 + FAISS + RRF)**: We implement a hybrid retrieval approach that combines BM25 and FAISS using Reciprocal Rank Fusion (RRF). Combining BM25 and FAISS leverages the strengths of both lexical and semantic search methods. BM25 is effective for keyword-based searches, while FAISS captures semantic relationships through vector embeddings. RRF allows us to fuse the results from both methods, improving overall retrieval performance.

Then, we apply Maximal Marginal Relevance (MMR) to enhance the quality of results by ensuring that the selected passages are not only relevant but also diverse, reducing redundancy and providing a broader perspective on the query.

The `hybrid_search` function in `fastapi_app/retrieval.py` implements this approach. 

### Methods

- **BM25**: A traditional information retrieval algorithm that ranks documents based on term frequency and inverse document frequency. It is effective for keyword-based searches.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors. It enables semantic search by comparing vector representations of text.
- **Reciprocal Rank Fusion (RRF)**: A method for combining multiple ranked lists into a single ranked list. It assigns scores to documents based on their ranks in the individual lists, allowing for a more robust ranking that leverages the strengths of each retrieval method.
- **Maximal Marginal Relevance (MMR)**: A technique for selecting a subset of documents that balances relevance to the query and diversity among the selected documents. It helps reduce redundancy in the results by promoting diverse content.

### Output
The output of the `hybrid_search` function is a list of the top-k passages that are both relevant and diverse, ready to be used for answer generation.

Sample output:
```json
{
  "query": "Where to park near Central Park?",
  "passages": [
    {"id":"wikivoyage_nyc_001#0", "title":"Central Park - Parking","url":"https://wikivoyage/centralpark","text":"Street parking is limited...","score":0.92},
    {"id":"wikivoyage_nyc_002#1", "title":"Parking Tips","url":"https://wikivoyage/parkingtips","text":"Consider using parking garages...","score":0.89}
  ],
  "meta":{"elapsed_ms":85.4,"used_k":2}
}
```

## 4) Answer Generation with Citations
The answer generation component synthesizes answers from the retrieved passages, incorporating citations to enhance credibility. 

It uses a language model to generate coherent responses, ensuring that each answer is supported by the relevant sources. To achieve this, we format the prompt to include the user's query alongside the retrieved passages, clearly indicating which passages should be cited if used in the answer. 

The generated answers include inline citations in a consistent format (e.g., [1], [2]), allowing users to trace back the information to its source. 

A self-correction mechanism is implemented to verify and refine the generated answers.

### Self-Correction Mechanism
To ensure the accuracy of the generated answers, we verify the generated answer against the retrieved passages and make adjustments if necessary. Scope for future improvement includes using RAGAS or similar techniques for faithfulness checking.

### Output
The output of the answer generation function is a structured response containing the generated answer and a list of citations. 

Sample output:
```json
{
  "query": "Where to park near Central Park?",
  "answer": "Street parking is limited around Central Park. It is recommended to use parking garages located nearby, such as the Central Parking Garage on 5th Ave [1]. Additionally, consider using public transportation to avoid parking hassles [2].",
  "citations": [
    {"id":"wikivoyage_nyc_001#0", "title":"Central Park - Parking","url":"https://wikivoyage/centralpark"},
    {"id":"wikivoyage_nyc_002#1", "title":"Parking Tips","url":"https://wikivoyage/parkingtips"}
  ],
  "meta":{"elapsed_ms":120.5,"used_k":2}
}
```

---

## Evaluation

To ensure the quality and reliability of the retrieval system, we implement both automatic and manual evaluation methods, allowing us to assess the performance of the system from multiple angles. The evaluation process loads seed QA, runs BM25/FAISS/Hybrid modes, and calculates Hits@k, MRR, NDCG. (RAGAS-based faithfulness checking is planned for future work.)

### Retrieval Evaluation

The evaluation script `eval/evaluate_retrieval.py` implements the evaluation logic. It assumes that the seed QA JSONL has at least the following fields:
- `question`: The query to be evaluated.
- `doc_contains`: A substring of the expected document's URL or title for matching.

#### Process & Metrics
The evaluation process involves running the specified retrieval method (`bm25`, `faiss`, or `hybrid`) for each question in the seed QA set. It checks if any of the top-k retrieved passages match the expected document based on the `doc_contains` field. Then, it computes the following metrics:
- Hits@k: Indicates whether at least one relevant document is found in the top-k results.
- MRR (Mean Reciprocal Rank): Measures the rank of the first relevant document in the retrieved list.
- NDCG@k (Normalized Discounted Cumulative Gain): Evaluates the quality of the ranking, considering the position of relevant documents.

The script aggregates these metrics across all questions and outputs the mean values, providing insights into the effectiveness of the retrieval method.

#### Usage

```bash
python eval/evaluate_retrieval.py --qa <path_to_qa_file> --mode <bm25|faiss|hybrid> --k <top_k> --pool <pool_size>
```

Alternatively:
```bash
python -m eval.evaluate_retrieval --qa <path_to_qa_file> --mode <bm25|faiss|hybrid> --k <top_k> --pool <pool_size>
```

This will output the evaluation metrics in JSON format and save the output to a file for later analysis 
If it's not saved, redirect it to a file (e.g., `> eval/runs/retrieval_<mode>.json`).

Run 3 modes:

```bash
# BM25 baseline
python -m eval.evaluate_retrieval --mode bm25 --k 5 --pool 50 --qa eval/fixtures/wikivoyage_nynj_qa.jsonl #> eval/runs/retrieval_bm25.json

# FAISS baseline
python -m eval.evaluate_retrieval --mode faiss --k 5 --pool 50 --qa eval/fixtures/wikivoyage_nynj_qa.jsonl #> eval/runs/retrieval_faiss.json

# Hybrid (BM25 + FAISS → RRF → MMR)
python -m eval.evaluate_retrieval --mode hybrid --k 5 --pool 50 --qa eval/fixtures/wikivoyage_nynj_qa.jsonl #> eval/runs/retrieval_hybrid.json
```

#### Acceptance Criteria
Acceptance for now: Hits@5 ≥ 0.6 on your small set (we’ll raise the bar after reranking).


### Answer Evaluation (WIP)

...

## Governance enforcement
Governance is implemented via regex-based filters that check both input queries and output answers for PII and toxic content. The governance checks are applied at two points in the RAG pipeline:

- Input checks are performed at the pipeline/API boundary (`fastapi_app.rag_pipeline.retrieve_only` `/ answer_query`).
- Output checks are performed inside the Answerer after LLM generation (`fastapi_app.answerer.synthesize_answer`).
- If the output is blocked, the Answerer returns a canonical error:
  `{"error": "governance_blocked", "reason": "<explanation>", "meta": {...}}`


## Observability & deployment (WIP)
Future work includes integrating Prometheus and Grafana for monitoring the FastAPI service. Deployment scripts and Docker configurations will be provided to facilitate easy deployment of the RAG system in various environments.

# Tooling & dependencies
- **Core stack:** Python 3.10, FastAPI, Pydantic, Uvicorn; retrieval via FAISS, SentenceTransformers, rank-bm25; governance via regex; LLM integration through OpenAI SDK hitting Ollama or custom endpoints; 
- Conda env locks major numeric libs (`numpy 1.26`, `faiss-cpu 1.8`, `scikit-learn 1.4`) to avoid ABI clashes.

