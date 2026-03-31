import unittest

from backend.models import (
    AddReelResponse,
    AgenticQueryResponse,
    CreateIngestionResponse,
    EnrichmentData,
    IngestionStatus,
    IngestionStatusResponse,
    QueryIntent,
    QueryMeta,
)


class AgenticSchemaTests(unittest.TestCase):
    def test_map_points_always_present(self):
        payload = AgenticQueryResponse(
            intent=QueryIntent.search,
            narrative="test",
            meta=QueryMeta(),
        )
        self.assertIsInstance(payload.map_points, list)
        self.assertEqual(payload.map_points, [])

    def test_cards_always_present(self):
        payload = AgenticQueryResponse(
            intent=QueryIntent.search,
            narrative="test",
            meta=QueryMeta(),
        )
        self.assertIsInstance(payload.cards, list)
        self.assertEqual(payload.cards, [])

    def test_ingestion_create_defaults_to_queued(self):
        payload = CreateIngestionResponse(job_id="job-1")
        self.assertEqual(payload.status, IngestionStatus.queued)

    def test_ingestion_status_allows_result_payload(self):
        result = AddReelResponse(
            reel_id="r1",
            summary="s",
            auto_tags=["food"],
            enrichment=EnrichmentData(summary="s"),
        )
        payload = IngestionStatusResponse(
            job_id="job-1",
            status=IngestionStatus.completed,
            stage="completed",
            reel_id="r1",
            result=result,
        )
        self.assertEqual(payload.result.reel_id, "r1")
        self.assertEqual(payload.status, IngestionStatus.completed)


if __name__ == "__main__":
    unittest.main()
