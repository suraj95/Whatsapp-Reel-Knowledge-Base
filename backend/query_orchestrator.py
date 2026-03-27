import logging
from typing import Any

from openai import OpenAI

from .handlers.recommendation_handler import handle_recommendation
from .handlers.search_handler import handle_search
from .handlers.trip_planner_handler import handle_trip_planning
from .helpers import embed_text
from .intent import detect_intent
from .models import AgenticQueryResponse, QueryIntent

logger = logging.getLogger(__name__)


def handle_query_agentic(query: str, top_k: int, client: OpenAI, index: Any) -> AgenticQueryResponse:
    intent_result = detect_intent(query=query, client=client)
    query_embedding = embed_text(client, query)

    if intent_result.intent == QueryIntent.trip_planning:
        response = handle_trip_planning(
            index=index,
            query_embedding=query_embedding,
            top_k=top_k,
            intent_result=intent_result,
            query=query,
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

    logger.info(
        "agentic_query route=%s intent=%s confidence=%.2f results=%d map_points=%d",
        response.meta.debug_route,
        response.intent.value,
        response.meta.confidence,
        response.meta.result_count,
        response.meta.map_points_count,
    )
    return response
