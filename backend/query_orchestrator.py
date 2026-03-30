import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .handlers.recommendation_handler import handle_recommendation
from .handlers.search_handler import handle_search
from .handlers.trip_planner_handler import handle_trip_planning
from .helpers import embed_text, format_conversational_query_response
from .intent import detect_intent
from .models import AgenticQueryResponse, QueryIntent

logger = logging.getLogger(__name__)

MIN_RESULT_SCORE = 0.35
# Cap trip planner + intent context string length (conversation + latest query).
TRIP_QUERY_MAX_CHARS = 8000


def _format_conversation_context(
    conversation_history: Optional[List[Dict[str, Any]]],
    max_chars: int = 4000,
) -> str:
    if not conversation_history:
        return ""
    chunks = []
    for m in conversation_history[-16:]:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if content:
            chunks.append(f"{role}: {content}")
    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _build_place_payload(response: AgenticQueryResponse):
    payload = []
    for source in response.sources:
        enrichment = source.enrichment
        payload.append(
            {
                "place_name": enrichment.place_name if enrichment else None,
                "city": enrichment.city if enrichment else None,
                "country": enrichment.country if enrichment else None,
                "category": enrichment.category if enrichment else None,
                "summary": enrichment.summary if enrichment else source.summary,
                "score": source.score,
            }
        )
    return payload


def _post_process_response(response: AgenticQueryResponse, query: str, client: OpenAI) -> AgenticQueryResponse:
    all_sources = response.sources or []
    kept_sources = [source for source in all_sources if source.score >= MIN_RESULT_SCORE]
    kept_reel_ids = {source.reel_id for source in kept_sources}
    kept_urls = {source.url for source in kept_sources}

    filtered_cards = []
    for card in response.cards:
        if card.reel_id:
            if card.reel_id in kept_reel_ids:
                filtered_cards.append(card)
            continue

        places = list((card.metadata or {}).get("places") or [])
        if places:
            kept_places = [p for p in places if p.get("source_url") in kept_urls]
            card.metadata["places"] = kept_places
            if kept_places:
                filtered_cards.append(card)
            continue

        filtered_cards.append(card)

    response.sources = kept_sources
    response.map_points = [point for point in response.map_points if point.reel_id in kept_reel_ids]
    response.cards = filtered_cards
    response.meta.result_count = len(response.sources)
    response.meta.map_points_count = len(response.map_points)
    response.meta.applied_filters["min_result_score"] = MIN_RESULT_SCORE
    response.meta.applied_filters["dropped_low_confidence"] = max(0, len(all_sources) - len(kept_sources))

    if not response.sources:
        if not response.meta.skip_conversational_rewrite:
            response.narrative = (
                "I found only low-confidence matches this time. Try rephrasing or save more reels. "
                "Do you want me to suggest a better query to find places faster?"
            )
        return response

    if response.meta.skip_conversational_rewrite:
        return response

    place_payload = _build_place_payload(response)
    try:
        narrative = format_conversational_query_response(
            client=client,
            query=query,
            places=place_payload,
        )
        if narrative:
            response.narrative = narrative
    except Exception:
        response.narrative = (
            f"I found {len(response.sources)} relevant places from your saved reels. "
            "Would you like help with how to get there or nearby places to stay?"
        )

    return response


def handle_query_agentic(
    query: str,
    top_k: int,
    client: OpenAI,
    index: Any,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> AgenticQueryResponse:
    context = _format_conversation_context(conversation_history)
    intent_result = detect_intent(
        query=query,
        client=client,
        conversation_context=context or None,
    )
    # Embed the current user message only so Pinecone matches stay stable and do not
    # balloon when prior turns mention many places.
    query_embedding = embed_text(client, query.strip())

    trip_query = query.strip()
    if context:
        trip_query = f"{context}\n{query}".strip()[-TRIP_QUERY_MAX_CHARS:]

    if intent_result.intent == QueryIntent.trip_planning:
        response = handle_trip_planning(
            index=index,
            query_embedding=query_embedding,
            top_k=top_k,
            intent_result=intent_result,
            query=trip_query,
            client=client,
        )
    elif intent_result.intent == QueryIntent.recommendation:
        response = handle_recommendation(
            index=index,
            query_embedding=query_embedding,
            top_k=top_k,
            intent_result=intent_result,
        )
    else:
        response = handle_search(
            index=index,
            query_embedding=query_embedding,
            top_k=top_k,
            intent_result=intent_result,
        )

    response = _post_process_response(response=response, query=query, client=client)

    logger.info(
        "agentic_query route=%s intent=%s confidence=%.2f results=%d map_points=%d",
        response.meta.debug_route,
        response.intent.value,
        response.meta.confidence,
        response.meta.result_count,
        response.meta.map_points_count,
    )
    return response
