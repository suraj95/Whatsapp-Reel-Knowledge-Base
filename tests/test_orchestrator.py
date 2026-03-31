import unittest
from unittest.mock import patch

from backend.models import (
    AgenticQueryResponse,
    EnrichmentData,
    MapPoint,
    QueryCard,
    QueryIntent,
    QueryMeta,
    ReelResult,
)
from backend.query_orchestrator import handle_query_agentic


def _response(
    intent: QueryIntent,
    route: str,
    sources=None,
    cards=None,
    map_points=None,
) -> AgenticQueryResponse:
    return AgenticQueryResponse(
        intent=intent,
        map_points=map_points or [],
        cards=cards or [],
        narrative="ok",
        sources=sources or [],
        meta=QueryMeta(debug_route=route),
    )


class QueryOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    @patch("backend.query_orchestrator.detect_intent")
    @patch("backend.query_orchestrator.embed_text")
    @patch("backend.query_orchestrator.handle_search")
    @patch("backend.query_orchestrator.format_conversational_query_response")
    async def test_search_route_dispatch(self, mock_formatter, mock_search, mock_embed, mock_detect):
        mock_detect.return_value = type(
            "IntentResult", (), {"intent": QueryIntent.search, "confidence": 0.9}
        )()
        mock_embed.return_value = [0.1, 0.2]
        mock_formatter.return_value = "Conversational reply"
        source = ReelResult(
            reel_id="r1",
            url="https://example.com/1",
            summary="Summary 1",
            auto_tags=[],
            score=0.81,
            enrichment=EnrichmentData(summary="E1", place_name="Place 1"),
        )
        mock_search.return_value = _response(
            QueryIntent.search,
            "search_handler",
            sources=[source],
        )

        res = await handle_query_agentic("query", 5, client=object(), index=object())
        self.assertEqual(res.meta.debug_route, "search_handler")
        self.assertEqual(res.intent, QueryIntent.search)
        self.assertEqual(res.narrative, "Conversational reply")

    @patch("backend.query_orchestrator.detect_intent")
    @patch("backend.query_orchestrator.embed_text")
    @patch("backend.query_orchestrator.handle_recommendation")
    @patch("backend.query_orchestrator.format_conversational_query_response")
    async def test_recommendation_route_dispatch(self, mock_formatter, mock_handler, mock_embed, mock_detect):
        mock_detect.return_value = type(
            "IntentResult", (), {"intent": QueryIntent.recommendation, "confidence": 0.9}
        )()
        mock_embed.return_value = [0.1, 0.2]
        mock_formatter.return_value = "Conversational reply"
        source = ReelResult(
            reel_id="r1",
            url="https://example.com/1",
            summary="Summary 1",
            auto_tags=[],
            score=0.81,
            enrichment=EnrichmentData(summary="E1", place_name="Place 1"),
        )
        mock_handler.return_value = _response(
            QueryIntent.recommendation,
            "recommendation_handler",
            sources=[source],
        )

        res = await handle_query_agentic("query", 5, client=object(), index=object())
        self.assertEqual(res.meta.debug_route, "recommendation_handler")
        self.assertEqual(res.intent, QueryIntent.recommendation)
        self.assertEqual(res.narrative, "Conversational reply")

    @patch("backend.query_orchestrator.detect_intent")
    @patch("backend.query_orchestrator.embed_text")
    @patch("backend.query_orchestrator.handle_trip_planning")
    @patch("backend.query_orchestrator.format_conversational_query_response")
    async def test_trip_planning_route_dispatch(self, mock_formatter, mock_handler, mock_embed, mock_detect):
        mock_detect.return_value = type(
            "IntentResult", (), {"intent": QueryIntent.trip_planning, "confidence": 0.9}
        )()
        mock_embed.return_value = [0.1, 0.2]
        mock_formatter.return_value = "Conversational reply"
        source = ReelResult(
            reel_id="r1",
            url="https://example.com/1",
            summary="Summary 1",
            auto_tags=[],
            score=0.81,
            enrichment=EnrichmentData(summary="E1", place_name="Place 1"),
        )
        mock_handler.return_value = _response(
            QueryIntent.trip_planning,
            "trip_planner_handler",
            sources=[source],
        )
        res = await handle_query_agentic("query", 5, client=object(), index=object())
        self.assertEqual(res.meta.debug_route, "trip_planner_handler")
        self.assertEqual(res.intent, QueryIntent.trip_planning)
        self.assertEqual(res.narrative, "Conversational reply")

    @patch("backend.query_orchestrator.detect_intent")
    @patch("backend.query_orchestrator.embed_text")
    @patch("backend.query_orchestrator.handle_search")
    @patch("backend.query_orchestrator.format_conversational_query_response")
    async def test_low_confidence_filter_prunes_output(
        self, mock_formatter, mock_search, mock_embed, mock_detect
    ):
        mock_detect.return_value = type(
            "IntentResult", (), {"intent": QueryIntent.search, "confidence": 0.9}
        )()
        mock_embed.return_value = [0.1, 0.2]
        mock_formatter.return_value = "Clean answer"

        high = ReelResult(
            reel_id="high-1",
            url="https://example.com/high-1",
            summary="High",
            auto_tags=[],
            score=0.8,
            enrichment=EnrichmentData(summary="High summary", place_name="High Place"),
        )
        low = ReelResult(
            reel_id="low-1",
            url="https://example.com/low-1",
            summary="Low",
            auto_tags=[],
            score=0.1,
            enrichment=EnrichmentData(summary="Low summary", place_name="Low Place"),
        )
        cards = [
            QueryCard(card_type="search_result", title="High card", reel_id="high-1"),
            QueryCard(card_type="search_result", title="Low card", reel_id="low-1"),
            QueryCard(
                card_type="itinerary_day",
                title="Day 1",
                metadata={
                    "places": [
                        {"place_name": "High Place", "source_url": "https://example.com/high-1"},
                        {"place_name": "Low Place", "source_url": "https://example.com/low-1"},
                    ]
                },
            ),
        ]
        map_points = [
            MapPoint(
                reel_id="high-1",
                place_name="High Place",
                source_url="https://example.com/high-1",
                score=0.8,
            ),
            MapPoint(
                reel_id="low-1",
                place_name="Low Place",
                source_url="https://example.com/low-1",
                score=0.1,
            ),
        ]

        mock_search.return_value = _response(
            QueryIntent.search,
            "search_handler",
            sources=[high, low],
            cards=cards,
            map_points=map_points,
        )

        res = await handle_query_agentic("query", 5, client=object(), index=object())
        self.assertEqual([s.reel_id for s in res.sources], ["high-1"])
        self.assertEqual([p.reel_id for p in res.map_points], ["high-1"])
        self.assertEqual(len(res.cards), 2)
        self.assertEqual(res.cards[1].metadata["places"][0]["place_name"], "High Place")
        self.assertEqual(res.meta.applied_filters["min_result_score"], 0.35)
        self.assertEqual(res.meta.applied_filters["dropped_low_confidence"], 1)
        self.assertEqual(res.narrative, "Clean answer")


    @patch("backend.query_orchestrator.detect_intent")
    @patch("backend.query_orchestrator.embed_text")
    @patch("backend.query_orchestrator.handle_trip_planning")
    @patch("backend.query_orchestrator.format_conversational_query_response")
    async def test_trip_skip_narrative(self, mock_formatter, mock_handler, mock_embed, mock_detect):
        mock_detect.return_value = type(
            "IntentResult", (), {"intent": QueryIntent.trip_planning, "confidence": 0.9}
        )()
        mock_embed.return_value = [0.1, 0.2]
        mock_formatter.return_value = "SHOULD_NOT_APPLY"
        res = _response(
            QueryIntent.trip_planning,
            "trip_planner_graph",
            sources=[],
        )
        res.meta.skip_conversational_rewrite = True
        res.narrative = "Planner narrative"
        mock_handler.return_value = res

        out = await handle_query_agentic("query", 5, client=object(), index=object())
        self.assertEqual(out.narrative, "Planner narrative")
        mock_formatter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
