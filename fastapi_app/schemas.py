from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class RetrieveRequest(BaseModel):
    q: str = Field(..., min_length=1)
    k: Optional[int] = Field(5, ge=1)

class Passage(BaseModel):
    id: str
    title: Optional[str] = None
    url: Optional[str] = None
    text: str
    score: Optional[float] = None

class RetrieveResponse(BaseModel):
    query: str
    passages: List[Passage]
    meta: Optional[dict] = None


class AnswerRequest(BaseModel):
    q: str
    k: Optional[int] = Field(5, ge=1)

class Citation(BaseModel):
    id: str
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None

class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    meta: Optional[dict] = None

class ErrorResponse(BaseModel):
    error: str
    reason: str
    meta: Optional[dict] = None