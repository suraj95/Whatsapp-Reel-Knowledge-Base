from ..models import AgenticQueryResponse, IntentDetectionResult, QueryMeta
from .common import build_map_points_and_count, build_search_cards, build_sources_from_matches, query_index


def handle_search(index, query_embedding, top_k: int, intent_result: IntentDetectionResult) -> AgenticQueryResponse:
    res = query_index(index=index, query_embedding=query_embedding, top_k=top_k)
    matches = res.matches or []
    sources = build_sources_from_matches(matches)
    map_points, geocoded_count = build_map_points_and_count(sources)
    cards = build_search_cards(sources)

    if not sources:
        narrative = "I could not find matching reels yet. Save a few reels first, then try again."
    else:
        narrative = f"Found {len(sources)} matching places from your saved reels."

    return AgenticQueryResponse(
        intent=intent_result.intent,
        map_points=map_points,
        cards=cards,
        narrative=narrative,
        sources=sources,
        meta=QueryMeta(
            confidence=intent_result.confidence,
            debug_route="search_handler",
            clarification_needed=intent_result.clarification_needed,
            geocoded_on_the_fly=geocoded_count,
            result_count=len(sources),
            map_points_count=len(map_points),
        ),
    )
