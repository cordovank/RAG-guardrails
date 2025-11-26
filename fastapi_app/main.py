from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
from time import perf_counter

# Local imports
from .schemas import RetrieveRequest, RetrieveResponse, AnswerRequest, AnswerResponse, ErrorResponse
from .rag_pipeline import answer_query, retrieve_only
from .settings import settings as app_settings
from .governance import enforce

# Prometheus metrics
LAT_RETR = Histogram("latency_retrieval_ms", "Retrieval latency (ms)")
LAT_ANS  = Histogram("latency_answer_ms", "Answer latency (ms)")
REQS     = Counter("requests_total", "Total requests", ["route"])


# FastAPI app
app = FastAPI(title="Prod RAG with Guardrails (scaffold)")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/retrieve", response_model=RetrieveResponse, responses={400: {"model": ErrorResponse}})
def retrieve(req: RetrieveRequest):
    REQS.labels("/retrieve").inc()

    # Validate k against app settings
    if req.k > app_settings.max_retrieved_docs:
        raise HTTPException(status_code=400, detail=f"k must be <= {app_settings.max_retrieved_docs}")

    t0 = perf_counter()
    out = retrieve_only(req.q, k=req.k)
    elapsed_ms = (perf_counter() - t0) * 1000
    LAT_RETR.observe(elapsed_ms)

    meta = {"elapsed_ms": elapsed_ms, "used_k": len(out.get("passages", []))}
    return RetrieveResponse(
        query=out["query"], 
        passages=out["passages"], 
        meta=meta
    )

@app.post("/answer", response_model=AnswerResponse, responses={400: {"model": ErrorResponse}})
def answer(req: AnswerRequest):
    REQS.labels("/answer").inc()

    # Validate k against app settings
    if req.k > app_settings.max_retrieved_docs:
        raise HTTPException(status_code=400, detail=f"k must be <= {app_settings.max_retrieved_docs}")
    
    # Ensure governance on input query
    try:
        enforce(input_txt=req.q, output_txt="")
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"governance_blocked: {str(e)}")

    t0 = perf_counter()

    try:
        out = answer_query(req.q, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"llm_error: {str(e)}")
    # Prefer LLM to provide length control, but enforce here if needed
    elapsed_ms = (perf_counter() - t0) * 1000
    LAT_ANS.observe(elapsed_ms)


    # Default meta from the pipeline (may be an error structure)
    meta = out.get("meta", {})
    meta.update({"elapsed_ms": elapsed_ms})

    # Internal pipeline error: structured validation error indicating no passages
    if isinstance(out, dict) and out.get("error") == "no_passages":
        # Return a polite, user-facing answer while preserving meta info
        return AnswerResponse(
            query=req.q,
            answer="I don't know.",
            citations=[],
            meta=meta.setdefault("reason", out.get("reason", "no_passages")),
        )

    # Normal path: assemble response from pipeline output
    return AnswerResponse(
        query=req.q,
        answer=out.get("answer", "") if isinstance(out, dict) else "",
        citations=out.get("citations", []) if isinstance(out, dict) else [],
        meta=meta,
    )
