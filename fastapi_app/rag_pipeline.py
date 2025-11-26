from .governance import enforce
from .retrieval import hybrid_search
from .answerer import synthesize_answer

from fastapi_app.settings import settings as set
# from .llm import ollama_openai as default, custom_llm_openai as custom

# App settings
DEFAULT_MAX_DOCS = set.max_retrieved_docs


def _cap_k(k: int) -> int:
    return min(k, DEFAULT_MAX_DOCS)


def retrieve_only(q: str, k: int = DEFAULT_MAX_DOCS):
    enforce(input_txt=q, output_txt="")
    return {
        "query": q,
        "passages": hybrid_search(q, k=_cap_k(k)),
    }


def answer_query(q: str, k: int = DEFAULT_MAX_DOCS):
    enforce(input_txt=q, output_txt="")

    # Retrieve passages (hybrid search)
    resp = retrieve_only(q, k)
    passages = resp.get("passages", [])

    # pass settings.enable_self_check etc if needed
    
    # Governance check (input + output). Will raise if blocked.
    # enforce(input_txt=q, output_txt=answer)

    answer = synthesize_answer(
        q, 
        passages,
        enable_self_check=set.enable_self_check,
        llm_choice=set.llm_choice,
        max_tokens=set.max_response_length
        )

    answer.get("meta", {}).update({"used_passages": len(passages)})
    return answer
