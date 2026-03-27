import unittest

from backend.models import AgenticQueryResponse, QueryIntent, QueryMeta


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


if __name__ == "__main__":
    unittest.main()
