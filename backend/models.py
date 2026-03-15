from typing import List, Optional

from pydantic import BaseModel


class AddReelRequest(BaseModel):
    url: str
    manual_tags: Optional[List[str]] = None


class AddReelResponse(BaseModel):
    reel_id: str
    summary: str
    auto_tags: List[str]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class ReelResult(BaseModel):
    reel_id: str
    url: str
    summary: str
    auto_tags: List[str]
    score: float


class QueryResponse(BaseModel):
    results: List[ReelResult]

