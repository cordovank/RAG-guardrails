"""
Answerer module:

Centralized answer generation logic for the RAG FastAPI application.
Answer generation + Citation formatting + Error handling + Self-Correction loop

Once input is provided (query + retrieved context), the answerer:
Steps:
1. Validates input and prepares context
- Assemble a context prompt from passages.
2. Generate answer using an LLM function (dependency-injectable).
- Format explicit citations (IDs) and a brief citations list.
3. Governance check on output (block if necessary)
4. (Optional) Self-correction: if citations are invalid, regenerate answer with corrected context
5. Output: final answer + citations
"""

from typing import Callable, Dict, List, Optional, Any, Tuple
import time
import logging
import json 
import re
from collections import OrderedDict

from .governance import enforce
from .llm import ollama_openai as default, custom_llm_openai as custom
from fastapi_app.settings import settings as set
from fastapi_app.checks import check_answer


# App settings
MAX_SNIPPET_CHARS = set.max_snippet_chars
MAX_LLM_RETRIES = 2  # number of retries for LLM calls


logger = logging.getLogger(__name__)


def _validate_inputs(query: str, passages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate inputs to synthesize_answer.

    Returns:
      - None if inputs are valid.
      - An ErrorResponse-like dict when validation fails. The shape follows
        the project's `ErrorResponse` Pydantic model: {"error": str, "reason": str, "meta": dict}
    """
    if not query or not isinstance(query, str):
        return {"error": "invalid_input", "reason": "query must be a non-empty string", "meta": {}}

    if passages is None:
        return {"error": "invalid_input", "reason": "passages must be provided (list expected)", "meta": {}}

    if not isinstance(passages, list):
        return {"error": "invalid_input", "reason": "passages must be a list", "meta": {"type": str(type(passages))}}

    if len(passages) == 0:
        # Consistent error shape for empty retrievals (caller can translate to a polite message)
        return {"error": "no_passages", "reason": "no passages retrieved for the query", "meta": {"reason_code": "no_passages"}}

    return None
    

def _get_snippet(text: str, max_snippet_chars: int = MAX_SNIPPET_CHARS) -> str:
    """Extract a snippet from the passage text, truncated to max_chars."""
    if not text:
        snippet = ""
    else:
        # Try to get the first sentence (naive split on period), else take prefix
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        snippet = sentences[0] if sentences else text[:max_snippet_chars]

        # truncate if too long
        if len(snippet) > max_snippet_chars:
                snippet = snippet[: max_snippet_chars - 3] + "..."

    return snippet
    

def _format_citations(passages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Build citation structures and a numeric citation map following a deterministic policy (numeric ids are assigned by passage order).

    A citation structure includes:
      - id: passage/document id
      - title: passage/document title
      - url: passage/document URL
      - snippet: a brief excerpt from the passage (truncated to max_snippet_chars)
      - score: passage relevance score
      - passage_index: index of the passage in the original passages list
      - url_fragment: URL fragment for the passage (if applicable)

    A citation map is a mapping of numeric strings to passage ids, e.g. {"1": "doc1#0", "2": "doc2#1"}

    Returns:
      - citation_struct: list of citation dicts (id, title, url, snippet, score, passage_index)
      - citation_map: mapping of numeric string -> passage id (e.g. {"1": "doc1#0", "2": "doc2#1"})
    """
    citation_struct: List[Dict[str, Any]] = []
    # parts: List[str] = []

    for i, p in enumerate(passages):
        pid = p.get("id") or f"generated:{i}"
        snippet = _get_snippet((p.get("text") or "") .strip())   
        url = p.get("url", "") 
        fragment = pid.split("#", 1)[1].strip() if "#" in pid else ""

        citation_entry = {
            "id": pid,
            "title": p.get("title", ""),
            "url": url,
            "snippet": snippet,
            "score": p.get("score", 0.0),
            "passage_index": i,
            "url_fragment": f"{url.rstrip('#')}#{fragment}" if url and fragment else pid
        }
        citation_struct.append(citation_entry)

        # Part of prompt context with numeric label for each citation
        # parts.append(f"[{cid}] Passage id={pid} title={citation_entry['title']}:\\n{snippet}\\n---")
    
    # Build ordered numeric citation map from the citation structure
    citation_map: Dict[str, str] = OrderedDict(
        (str(i + 1), entry["url_fragment"]) for i, entry in enumerate(citation_struct)
    )

    return citation_struct, citation_map


def _build_context(citations_info: Tuple[List[Dict[str, Any]], Dict[str, str]]) -> str:
    """Build a compact context from citations.

    Example Format:
      [1] Passage id=doc1#0:
      This is a snippet from the first passage.
      ---

    Returns:
      A string suitable for inclusion in the LLM prompt.
    """
    citation_struct, citation_map = citations_info
    parts = []
    
    for idx, entry in enumerate(citation_struct, start=1):
        cid = str(idx)
        parts.append(
            f"[{cid}] Passage id={entry['id']}:\n"
            f"{entry['snippet']}\n---"
        )

    return "\n".join(parts)


def _build_prompt(context: str, query: str, max_tokens: int) -> str:
    """ Build a prompt that forces:
    - use ONLY provided CONTEXT
    - numeric citations [1], [2], ...
    - abstain with "I don't know."
    - JSON-only output: {"answer": "...", "used_citation_numbers": ["1","2"]}
    """
    system_str = (
        "You are an assistant that MUST answer the user's QUESTION using ONLY the provided CONTEXT PASSAGES.\n"
        "If the CONTEXT does NOT contain the information needed to answer, you MUST respond: \"I don't know.\"\n"
        "Do NOT use external knowledge. "
        "Do NOT fabricate facts, reasoning, or citations."
    )

    instruct = (
        "Instructions:\n"
        "1. Respond concisely using ONLY information found in the CONTEXT.\n"
        "2. Cite supporting passages using numeric inline citations like [1], [2].\n"
        "3. Place citations immediately after the sentence they support.\n"
        "4. Use ONLY citations for passages that actually contain the referenced information.\n"
        "5. If the answer is not fully supported by the CONTEXT, respond EXACTLY: \"I don't know.\"\n"
        "6. Do NOT invent citations, details, or assumptions.\n"
        f"7. Limit the final answer to {max_tokens} characters if longer."
    )

    return (
        system_str
        + "\n\nCONTEXT PASSAGES:\n"
        + context
        + "\n\nQUESTION:\n"
        + query
        + "\n\n"
        + instruct
    )


def _prepare_message_content(citation_info: Tuple[List[Dict[str, Any]], Dict[str, str]], query: str, max_tokens: int) -> str:
    """Prepares the message content for the user role in the LLM chat.

    The message content includes the formatted context and the prompt.
    It is used to instruct the LLM on how to respond to the user's query.

    Returns (prompt, citation_text, citations_struct, citation_map)
    """
    ctx = _build_context(citation_info)
    prompt = _build_prompt(ctx, query, max_tokens)
    return prompt


def _parse_numeric_citations(answer_text: str, citation_map: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Parse numeric citations like [1] and map to passage ids. Returns (used_nums, used_passage_ids)."""
    used_nums: List[str] = []
    used_passage_ids: List[str] = []

    if not answer_text:
        return used_nums, used_passage_ids
    
    try:
        found = re.findall(r"\[([0-9]+)\]", answer_text)
        for f in found:
            if f not in used_nums:
                used_nums.append(f)
                pid = citation_map.get(f)
                # Verify pid exists
                if pid:
                    used_passage_ids.append(pid)
                else:
                    logger.debug("LLM referenced unknown citation number [%s]", f)

    except Exception:
        logger.debug("Failed to parse citations from LLM output", exc_info=True)
    return used_nums, used_passage_ids


def _build_response(final_answer: str, citation_info: Tuple[List[Dict[str, Any]], Dict[str, str]], resp_limit: int) -> Dict[str, Any]:
    """Assemble the final response dict with meta and citations."""
    # human readable mapping text
    citations_struct, citation_map = citation_info
    used_cids, used_pids = _parse_numeric_citations(final_answer, citation_map)
    citation_map_str = "Citations: \n" + "\n - ".join(f"[{cid}] Passage: {pid}" for cid, pid in citation_map.items())

    # Truncate long answers as a final safety net
    if resp_limit and len(final_answer) > resp_limit:
        final_answer = final_answer[: resp_limit - 3] + "..."

    # Metadata
    meta = {}
    try:
        meta["citation_map"] = citation_map
        meta["citation_text"] = citation_map_str
        meta["used_citation_count"] = len(used_cids)
        meta["used_citation_numbers"] = used_cids
        meta["used_passage_ids"] = used_pids
    except Exception:
        logger.debug("Failed to attach citation metadata", exc_info=True)

    return {
        "answer": final_answer,
        "citations": citations_struct,
        "meta": meta,
    }


def self_check_answer(raw_answer, citation_info, enable_self_check: bool, llm_fn: Optional[Callable[[str], str]] = None) -> str:
    """
    Self-check mechanism to validate and potentially correct the generated answer.
    If enable_self_check is True, the provided check_fn is called with (query, answer, passages).
    If check_fn returns a non-empty corrected answer different from raw_answer, it is used as
    the final answer.

    Returns the final answer (either original or corrected).
    """
    final_answer = raw_answer
    if enable_self_check:
        logger.debug("Running self-check on answer")
        try:
            output = check_answer(raw_answer, citation_info, llm_fn)
            corrected_ans, meta = output
            logger.debug("Self-check result: corrected=%s, meta=%s", corrected_ans, meta)
            if corrected_ans and isinstance(corrected_ans, str) and corrected_ans.strip() and corrected_ans.strip() != raw_answer.strip():
                final_answer = corrected_ans
        except Exception:
            logger.debug("Self-check failed; proceeding with original answer", exc_info=True)

    return final_answer


def synthesize_answer(
    query: str,
    passages: List[Dict[str, Any]],
    enable_self_check: bool = False,
    llm_choice: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Synthesize an answer from the query and retrieved passages.

    Inputs:
      - query: user question
      - passages: list of passage dicts (must include 'id' and 'text')
      - enable_self_check: if True, run check_fn to optionally correct the answer
      - llm_choice: "ollama" or "custom" to select the LLM backend
      - max_tokens: max tokens/characters for the LLM response

    Returns a dict with either 'answer' and 'citations' or an 'error' object.
    """

    # 1. Validate inputs
    validation_result = _validate_inputs(query, passages)
    if validation_result is not None:
        return validation_result

    # 2. Format citations from passages
    citation_info = _format_citations(passages)

    # 3. Prepare prompt/context and invoke LLM
    prompt = _prepare_message_content(citation_info, query, max_tokens)
    llm_fn = default.get_llm_callable() if llm_choice == "ollama" else custom.get_llm_callable()
    raw_answer = llm_fn(prompt=prompt, max_tokens=max_tokens)

    # 4. Optional self-check (non-fatal)
    final_answer = self_check_answer(raw_answer, citation_info, enable_self_check, llm_fn)

    # 5. Governance enforcement on LLM output (may raise or return error)
    try:
        enforce(input_txt=query, output_txt=final_answer)
    except Exception as e:
        logger.info("Governance blocked the answer: %s", e)
        return {"error": "governance_blocked", "reason": str(e), "meta": {}}

    # 6. Assemble and return final response
    return _build_response(final_answer, citation_info, resp_limit=max_tokens)
