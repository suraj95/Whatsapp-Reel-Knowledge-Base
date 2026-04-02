from typing import Any, List

from openai import OpenAI

from ..graphs.trip_planner_graph import run_trip_planner_graph
from ..models import AgenticQueryResponse, IntentDetectionResult, QueryMeta
from ..observability import get_log


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
        get_log().exception(
            "trip_planner_graph_failed",
            agent="trip_planner_handler",
            error_type=type(ex).__name__,
            error_message=str(ex)[:500],
        )
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
