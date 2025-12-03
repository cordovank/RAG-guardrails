"""
Conservative self-check function for validating generated answers against retrieved passages.
Intended to be used with the answerer.synthesize_answer's enable_self_check and check_fn parameters.
Returns either a corrected answer or None if unable to auto-repair.
"""

from typing import List, Dict, Optional, Callable, Tuple, Any
import re
import logging
from collections import OrderedDict
import string
import difflib

logger = logging.getLogger(__name__)


def _extract_numeric_citations(text: str) -> List[str]:
    """Return unique citation numbers in order of appearance."""
    return list(dict.fromkeys(re.findall(r"\[([0-9]+)\]", text)))


def _strip_unknown_citation_nums(text: str, valid_nums: set) -> Tuple[str, bool]:
    """
    Remove bracketed citation numbers that are not in valid_nums.
    Returns (cleaned_text, changed_flag)
    """
    changed = False

    def repl(match):
        nonlocal changed
        num = match.group(1)
        if num in valid_nums:
            return f"[{num}]"
        else:
            changed = True
            return ""

    cleaned = re.sub(r"\[([0-9]+)\]", repl, text) # Remove unknown citations
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip() # Clean up extra spaces
    return cleaned, changed


def _call_llm_edit(answer: str, missing_nums: List[str], citation_map: Dict[str, str], passage_snippets: Dict[str, str], llm_fn: Callable[[str], str]) -> Optional[str]:
    """
    Call the provided llm_fn with a short edit prompt. The llm_fn should accept a single string prompt and return the edited answer text.
    Returns edited text or None on failure.
    """
    try:
        numbers_text = ", ".join(missing_nums)
        prompt = (
            "You are an editing assistant.\n"
            "Your task: Insert ONLY the missing citation numbers where their referenced "
            "passage content appears in the given ANSWER.\n\n"

            "Rules (MANDATORY):\n"
            "1. Preserve the original wording of the ANSWER.\n"
            "2. Add only the specified missing numeric citations: [" + numbers_text + "].\n"
            "3. Use citation format [n] with no extra text.\n"
            "4. Place a citation immediately after the sentence or clause supported by the passage.\n"
            "5. Do NOT change meaning, reorder sentences, or add new information.\n"
            "6. Do NOT output explanations, notes, markdown, or JSON.\n"
            "7. Output ONLY the corrected answer text—no preamble, no trailing commentary.\n\n"

            "ANSWER:\n"
            f"{answer}\n\n"

            "CITATION MAP (use these to identify where citations belong):\n"
        )
        
        for num, pid in citation_map.items():
            snippet = passage_snippets.get(num, "")
            prompt += f"[{num}] — {pid} — snippet: {snippet}\n"

        prompt += (
            "\nReturn the corrected ANSWER text only."
        )

        logger.debug("LLM edit prompt:\n%s", prompt)
        
        edited = llm_fn(prompt)
        if not edited or not isinstance(edited, str):
            return None
        
        return edited.strip()
    
    except Exception:
        logger.exception("LLM edit call failed")
        return None


def check_answer(
    answer: str,
    citation_info: Tuple[List[Dict[str, Any]], Dict[str, str]],
    llm_fn: Optional[Callable[[str], str]] = None,
    fuzzy_threshold: float = 0.75,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Main check function that returns (corrected_answer_or_None, meta).

    Parameters:
      - answer: raw LLM answer text
      - citation_info: tuple (citation_struct_list, citation_map) as produced by _format_citations
      - llm_edit_fn: optional function that accepts a prompt string and returns edited answer text
      - fuzzy_threshold: threshold for fuzzy matching

    Meta keys returned:
      - auto_repaired: bool
      - repair_strategy: str
      - missing_before: int
      - missing_after: int
      - details: optional dict with more debug info
    """
    meta: Dict[str, Any] = {
        "auto_repaired": False,
        "repair_strategy": None,
        "missing_before": 0,
        "missing_after": 0,
        "details": {},
    }
    
    if not answer or not isinstance(answer, str):
        return None, meta

    # Prep passage snippets for citation numbers
    citation_struct, citation_map = citation_info
    passage_snippets = {str(i + 1): (entry.get("snippet") or "") for i, entry in enumerate(citation_struct)}
    all_nums = list(citation_map.keys())

    # Clean answer by stripping unknown/invalid citations & identify missing citations
    cleaned_answer, is_unknown_removed = _strip_unknown_citation_nums(answer, set(all_nums))
    used_before = _extract_numeric_citations(cleaned_answer)
    missing_nums = [n for n in all_nums if n not in used_before]

    meta["missing_before"] = len(missing_nums)
    meta["used_before"] = used_before

    # If no missing citations and none were stripped, return as is
    if not missing_nums:
        if is_unknown_removed:
            meta.update({"auto_repaired": True, "repair_strategy": "strip_unknowns", "missing_after": 0, "used_after": used_before})
            return cleaned_answer, meta
        else:
            meta.update({"auto_repaired": False, "repair_strategy": "none", "missing_after": 0, "used_after": used_before})
            return cleaned_answer, meta

    # LLM edit pass
    if missing_nums and llm_fn is not None:
        try:
            edited = _call_llm_edit(cleaned_answer, missing_nums, citation_map, passage_snippets, llm_fn)
            if edited:
                used_after = _extract_numeric_citations(edited)
                meta["used_after"] = used_after
                return edited, meta
        except Exception:
            logger.exception("LLM edit failed")
    return None, meta
