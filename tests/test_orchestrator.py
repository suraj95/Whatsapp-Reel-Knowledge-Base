import unittest
from unittest.mock import patch

from backend.models import AgenticQueryResponse, QueryIntent, QueryMeta
from backend.query_orchestrator import handle_query_agentic


def _response(intent: QueryIntent, route: str) -> AgenticQueryResponse:
    return AgenticQueryResponse(
        intent=intent,
        map_points=[],
        cards=[],
        narrative="ok",
        sources=[],
        meta=QueryMeta(debug_route=route),
    )


class QueryOrchestratorTests(unittest.TestCase):
    @patch("backend.query_orchestrator.detect_intent")
    @patch("backend.query_orchestrator.embed_text")
    @patch("backend.query_orchestrator.handle_search")
    def test_search_route_dispatch(self, mock_search, mock_embed, mock_detect):
        mock_detect.return_value = type(
            "IntentResult", (), {"intent": QueryIntent.search, "confidence": 0.9}
        )()
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = _response(QueryIntent.search, "search_handler")

        res = handle_query_agentic("query", 5, client=object(), index=object())
        self.assertEqual(res.meta.debug_route, "search_handler")
        self.assertEqual(res.intent, QueryIntent.search)

    @patch("backend.query_orchestrator.detect_intent")
    @patch("backend.query_orchestrator.embed_text")
    @patch("backend.query_orchestrator.handle_recommendation")
    def test_recommendation_route_dispatch(self, mock_handler, mock_embed, mock_detect):
        mock_detect.return_value = type(
            "IntentResult", (), {"intent": QueryIntent.recommendation, "confidence": 0.9}
        )()
        mock_embed.return_value = [0.1, 0.2]
        mock_handler.return_value = _response(QueryIntent.recommendation, "recommendation_handler")

        res = handle_query_agentic("query", 5, client=object(), index=object())
        self.assertEqual(res.meta.debug_route, "recommendation_handler")
        self.assertEqual(res.intent, QueryIntent.recommendation)

    @patch("backend.query_orchestrator.detect_intent")
    @patch("backend.query_orchestrator.embed_text")
    @patch("backend.query_orchestrator.handle_trip_planning")
    def test_trip_planning_route_dispatch(self, mock_handler, mock_embed, mock_detect):
        mock_detect.return_value = type(
            "IntentResult", (), {"intent": QueryIntent.trip_planning, "confidence": 0.9}
        )()
        mock_embed.return_value = [0.1, 0.2]
        mock_handler.return_value = _response(QueryIntent.trip_planning, "trip_planner_handler")

        res = handle_query_agentic("query", 5, client=object(), index=object())
        self.assertEqual(res.meta.debug_route, "trip_planner_handler")
        self.assertEqual(res.intent, QueryIntent.trip_planning)


if __name__ == "__main__":
    unittest.main()
