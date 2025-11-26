"""
Governance enforcement helper.

Contract:
- enforce(input_txt, output_txt) validates the combined input/output pair and raises on block.
- Enforcement responsibilities in this project:
  - Pipeline/API boundary: input validation (fastapi_app.rag_pipeline.*).
  - Answerer: output validation after LLM generation (fastapi_app.answerer.synthesize_answer).
"""

import re
from typing import Dict

PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),   # SSN
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
]
BAD_WORDS = {"hate", "kill"}  # Expand as needed

def scan_text(s: str) -> Dict[str, bool]:
    pii = any(p.search(s) for p in PII_PATTERNS)
    tox = any(w in s.lower() for w in BAD_WORDS)
    return {"pii": pii, "toxicity": tox}

def enforce(input_txt: str, output_txt: str) -> None:
    issues = []
    for label, flag in scan_text(input_txt).items():
        if flag: issues.append(f"input_{label}")
    for label, flag in scan_text(output_txt).items():
        if flag: issues.append(f"output_{label}")
    if issues:
        raise ValueError(f"Governance gate failure: {', '.join(issues)}")
