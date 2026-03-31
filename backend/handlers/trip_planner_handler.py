import logging
from typing import Any, List

from openai import OpenAI

from ..graphs.trip_planner_graph import run_trip_planner_graph
from ..models import AgenticQueryResponse, IntentDetectionResult, QueryMeta

logger = logging.getLogger(__name__)


async def handle_trip_planning(
    index: Any,
    query_embedding: List[float],
    top_k: int,
    intent_result: IntentDetectionResult,
    query: str,
    client: OpenAI,
) -> AgenticQueryResponse:
    try:
        return await run_trip_planner_graph(
            query=query,
            index=index,
            query_embedding=query_embedding,
            top_k=top_k,
            intent_result=intent_result,
            client=client,
        )
    except Exception as ex:
        logger.exception("trip_planner_graph failed: %s", ex)
        return AgenticQueryResponse(
            intent=intent_result.intent,
            map_points=[],
            cards=[],
            narrative=f"Trip planner hit an error: {ex}",
            sources=[],
            meta=QueryMeta(
                confidence=intent_result.confidence,
                debug_route="trip_planner_handler_error",
                clarification_needed=True,
                applied_filters={"error": str(ex)},
            ),
        )
