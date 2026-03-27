from typing import Any, Dict, List, Tuple

from ..helpers import _approximate_coordinates, strip_igsh_parameter
from ..models import EnrichmentData, MapPoint, QueryCard, ReelResult


def _to_enrichment(meta: Dict) -> EnrichmentData:
    return EnrichmentData(
        place_name=meta.get("enrichment_place_name"),
        city=meta.get("enrichment_city"),
        country=meta.get("enrichment_country"),
        lat=meta.get("enrichment_lat"),
        lng=meta.get("enrichment_lng"),
        rating=meta.get("enrichment_rating"),
        category=meta.get("enrichment_category"),
        cuisine=meta.get("enrichment_cuisine"),
        price_range=meta.get("enrichment_price_range"),
        summary=meta.get("enrichment_summary", meta.get("summary", "")),
        tags=meta.get("enrichment_tags", []),
    )


def query_index(index: Any, query_embedding: List[float], top_k: int):
    return index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )


def build_sources_from_matches(matches) -> List[ReelResult]:
    sources: List[ReelResult] = []
    for match in matches:
        meta = match.metadata or {}
        score = float(match.score) if match.score is not None else 0.0
        enrichment = _to_enrichment(meta)
        sources.append(
            ReelResult(
                reel_id=match.id,
                url=strip_igsh_parameter(meta.get("url", "")),
                summary=meta.get("summary", ""),
                auto_tags=meta.get("tags", []),
                score=score,
                enrichment=enrichment,
            )
        )
    return sources


def build_map_points_and_count(sources: List[ReelResult]) -> Tuple[List[MapPoint], int]:
    map_points: List[MapPoint] = []
    geocoded_count = 0

    for source in sources:
        enrichment = source.enrichment
        if not enrichment:
            continue

        place_name = enrichment.place_name or "Unknown place"
        lat = enrichment.lat
        lng = enrichment.lng
        city = enrichment.city
        country = enrichment.country

        if lat is None or lng is None:
            approx_lat, approx_lng, approx_city, approx_country = _approximate_coordinates(
                place_name=enrichment.place_name,
                city=city,
                country=country,
            )
            if approx_lat is not None and approx_lng is not None:
                geocoded_count += 1
                lat = approx_lat
                lng = approx_lng
                city = city or approx_city
                country = country or approx_country

        map_points.append(
            MapPoint(
                reel_id=source.reel_id,
                place_name=place_name,
                lat=lat,
                lng=lng,
                city=city,
                country=country,
                category=enrichment.category,
                score=source.score,
                source_url=source.url,
            )
        )
    return map_points, geocoded_count


def build_search_cards(sources: List[ReelResult]) -> List[QueryCard]:
    cards: List[QueryCard] = []
    for source in sources:
        enrichment = source.enrichment
        subtitle = ", ".join(
            [x for x in [enrichment.city if enrichment else None, enrichment.country if enrichment else None] if x]
        )
        cards.append(
            QueryCard(
                card_type="search_result",
                title=enrichment.place_name if enrichment and enrichment.place_name else "Unknown place",
                subtitle=subtitle or None,
                summary=enrichment.summary if enrichment else source.summary,
                reel_id=source.reel_id,
                source_url=source.url,
                score=source.score,
            )
        )
    return cards
