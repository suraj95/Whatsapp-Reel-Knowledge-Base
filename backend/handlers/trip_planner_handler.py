import re
from typing import Dict, List

from ..models import AgenticQueryResponse, IntentDetectionResult, QueryCard, QueryMeta
from .common import build_map_points_and_count, build_sources_from_matches, query_index


def _extract_trip_days(query: str) -> int:
    lowered = (query or "").lower()
    m = re.search(r"(\d+)\s*day", lowered)
    if not m:
        return 2
    try:
        value = int(m.group(1))
        return max(1, min(7, value))
    except Exception:
        return 2


def _build_day_buckets(sources, days: int) -> Dict[int, list]:
    buckets: Dict[int, list] = {day: [] for day in range(1, days + 1)}
    if not sources:
        return buckets
    i = 0
    for source in sources:
        day = (i % days) + 1
        buckets[day].append(source)
        i += 1
    return buckets


def handle_trip_planning(index, query_embedding, top_k: int, intent_result: IntentDetectionResult, query: str) -> AgenticQueryResponse:
    days = _extract_trip_days(query)
    res = query_index(index=index, query_embedding=query_embedding, top_k=max(top_k * 2, top_k))
    matches = res.matches or []
    sources = build_sources_from_matches(matches)

    # Keep top sources for plan compactness.
    selected = sources[: max(days * 3, top_k)]
    day_buckets = _build_day_buckets(selected, days)

    map_points, geocoded_count = build_map_points_and_count(selected)
    # Annotate day + sequence for map-first day-wise output.
    reel_to_day_sequence = {}
    for day in range(1, days + 1):
        seq = 1
        for source in day_buckets[day]:
            reel_to_day_sequence[source.reel_id] = (f"Day {day}", seq)
            seq += 1
    for point in map_points:
        group, sequence = reel_to_day_sequence.get(point.reel_id, ("Day 1", 1))
        point.group = group
        point.sequence = sequence

    cards: List[QueryCard] = []
    for day in range(1, days + 1):
        items = day_buckets[day]
        if not items:
            cards.append(
                QueryCard(
                    card_type="itinerary_day",
                    title=f"Day {day}",
                    summary="No strong matches yet for this day. Try adding more reels for better planning.",
                    metadata={"places": []},
                )
            )
            continue
        places = []
        for source in items:
            enrichment = source.enrichment
            places.append(
                {
                    "place_name": enrichment.place_name if enrichment else "Unknown place",
                    "city": enrichment.city if enrichment else None,
                    "country": enrichment.country if enrichment else None,
                    "source_url": source.url,
                }
            )
        cards.append(
            QueryCard(
                card_type="itinerary_day",
                title=f"Day {day}",
                summary=f"{len(items)} stop(s) planned from your saved reels.",
                metadata={"places": places},
            )
        )

    narrative = (
        f"Built a {days}-day draft itinerary from your saved reels. Map markers are grouped day-wise."
        if selected
        else "I could not find enough places to build an itinerary yet. Save more reels and try again."
    )

    return AgenticQueryResponse(
        intent=intent_result.intent,
        map_points=map_points,
        cards=cards,
        narrative=narrative,
        sources=selected,
        meta=QueryMeta(
            confidence=intent_result.confidence,
            debug_route="trip_planner_handler",
            clarification_needed=intent_result.clarification_needed,
            geocoded_on_the_fly=geocoded_count,
            result_count=len(selected),
            map_points_count=len(map_points),
            applied_filters={"trip_days": days},
        ),
    )
