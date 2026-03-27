from collections import defaultdict
from typing import List

from ..models import AgenticQueryResponse, IntentDetectionResult, MapPoint, QueryCard, QueryMeta, ReelResult
from .common import build_map_points_and_count, build_sources_from_matches, query_index


def _diverse_top_k(sources: List[ReelResult], top_k: int) -> List[ReelResult]:
    """
    Simple diversity pass: avoid over-concentrating on one category/city.
    """
    selected: List[ReelResult] = []
    city_count = defaultdict(int)
    category_count = defaultdict(int)

    for source in sources:
        if len(selected) >= top_k:
            break
        enrichment = source.enrichment
        city = (enrichment.city if enrichment else None) or "unknown_city"
        category = (enrichment.category if enrichment else None) or "unknown_category"

        if city_count[city] >= 2:
            continue
        if category_count[category] >= 2:
            continue

        selected.append(source)
        city_count[city] += 1
        category_count[category] += 1

    # Backfill if we filtered too aggressively.
    if len(selected) < top_k:
        selected_ids = {s.reel_id for s in selected}
        for source in sources:
            if source.reel_id in selected_ids:
                continue
            selected.append(source)
            if len(selected) >= top_k:
                break

    return selected


def handle_recommendation(index, query_embedding, top_k: int, intent_result: IntentDetectionResult) -> AgenticQueryResponse:
    res = query_index(index=index, query_embedding=query_embedding, top_k=max(top_k * 3, top_k))
    matches = res.matches or []
    all_sources = build_sources_from_matches(matches)
    selected_sources = _diverse_top_k(all_sources, top_k=top_k)

    map_points, geocoded_count = build_map_points_and_count(selected_sources)
    for point in map_points:
        point.group = point.category or "general"

    cards: List[QueryCard] = []
    for source in selected_sources:
        enrichment = source.enrichment
        location = ", ".join([x for x in [enrichment.city if enrichment else None, enrichment.country if enrichment else None] if x])
        cards.append(
            QueryCard(
                card_type="recommendation",
                title=enrichment.place_name if enrichment and enrichment.place_name else "Recommended place",
                subtitle=location or None,
                summary=(enrichment.summary if enrichment else source.summary),
                reason="Relevant to your query and diversified across your saved places.",
                reel_id=source.reel_id,
                source_url=source.url,
                score=source.score,
            )
        )

    narrative = (
        "Top picks based on your saved reels, balanced across cities and categories."
        if selected_sources
        else "I could not find recommendation candidates yet. Save more reels and try again."
    )

    return AgenticQueryResponse(
        intent=intent_result.intent,
        map_points=map_points,
        cards=cards,
        narrative=narrative,
        sources=selected_sources,
        meta=QueryMeta(
            confidence=intent_result.confidence,
            debug_route="recommendation_handler",
            clarification_needed=intent_result.clarification_needed,
            geocoded_on_the_fly=geocoded_count,
            result_count=len(selected_sources),
            map_points_count=len(map_points),
        ),
    )
