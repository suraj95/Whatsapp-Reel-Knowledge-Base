from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class QueryIntent(str, Enum):
    trip_planning = "trip_planning"
    search = "search"
    recommendation = "recommendation"
    unknown = "unknown"


class IntentEntities(BaseModel):
    destination: Optional[str] = None
    dates: Optional[str] = None
    budget: Optional[str] = None
    trip_length: Optional[str] = None
    food_pref: Optional[str] = None


class IntentDetectionResult(BaseModel):
    intent: QueryIntent
    confidence: float = 0.0
    entities: IntentEntities = Field(default_factory=IntentEntities)
    clarification_needed: bool = False
    reason: Optional[str] = None


class MapPoint(BaseModel):
    reel_id: str
    place_name: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    city: Optional[str] = None
    country: Optional[str] = None
    category: Optional[str] = None
    score: float = 0.0
    source_url: str
    group: Optional[str] = None
    sequence: Optional[int] = None


class QueryCard(BaseModel):
    card_type: str
    title: str
    subtitle: Optional[str] = None
    summary: Optional[str] = None
    reason: Optional[str] = None
    reel_id: Optional[str] = None
    source_url: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryMeta(BaseModel):
    confidence: float = 0.0
    applied_filters: Dict[str, Any] = Field(default_factory=dict)
    debug_route: str = "search_handler"
    clarification_needed: bool = False
    geocoded_on_the_fly: int = 0
    result_count: int = 0
    map_points_count: int = 0


class AgenticQueryResponse(BaseModel):
    intent: QueryIntent
    map_points: List[MapPoint] = Field(default_factory=list)
    cards: List[QueryCard] = Field(default_factory=list)
    narrative: str
    sources: List[ReelResult] = Field(default_factory=list)
    meta: QueryMeta

