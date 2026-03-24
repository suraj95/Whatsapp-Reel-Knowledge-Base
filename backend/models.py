from typing import List, Optional

from pydantic import BaseModel


class EnrichmentData(BaseModel):
    place_name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    rating: Optional[float] = None
    category: Optional[str] = None
    cuisine: Optional[str] = None
    price_range: Optional[str] = None
    summary: str
    tags: List[str] = []


class AddReelRequest(BaseModel):
    url: str
    manual_tags: Optional[List[str]] = None


class AddReelResponse(BaseModel):
    reel_id: str
    summary: str
    auto_tags: List[str]
    enrichment: Optional[EnrichmentData] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class ReelResult(BaseModel):
    reel_id: str
    url: str
    summary: str
    auto_tags: List[str]
    score: float
    enrichment: Optional[EnrichmentData] = None


class QueryResponse(BaseModel):
    results: List[ReelResult]


class EnrichRequest(BaseModel):
    vision_summary: str


class EnrichResponse(BaseModel):
    enrichment: EnrichmentData

