import unittest

from backend.intent import detect_intent
from backend.models import QueryIntent


class _FailingClient:
    class _Chat:
        class _Completions:
            @staticmethod
            def create(*args, **kwargs):
                raise RuntimeError("No network in tests")

        completions = _Completions()

    chat = _Chat()


class IntentDetectionTests(unittest.TestCase):
    def test_trip_planning_rule(self):
        result = detect_intent("Plan a 3 day Goa itinerary", _FailingClient())
        self.assertEqual(result.intent, QueryIntent.trip_planning)
        self.assertGreaterEqual(result.confidence, 0.86)

    def test_recommendation_rule(self):
        result = detect_intent("Recommend best cafes in Bali", _FailingClient())
        self.assertEqual(result.intent, QueryIntent.recommendation)
        self.assertGreaterEqual(result.confidence, 0.86)

    def test_default_search_rule(self):
        result = detect_intent("show restaurants we saved in goa", _FailingClient())
        self.assertEqual(result.intent, QueryIntent.search)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_trip_followup_reach_there(self):
        result = detect_intent("How do I reach there?", _FailingClient())
        self.assertEqual(result.intent, QueryIntent.trip_planning)

    def test_trip_top_ways_to_reach_not_recommendation(self):
        result = detect_intent("top ways to reach Goa", _FailingClient())
        self.assertEqual(result.intent, QueryIntent.trip_planning)

    def test_planning_word_not_just_plan(self):
        # "planning" must match; old \bplan\b missed this.
        result = detect_intent("We're planning a weekend in Paris", _FailingClient())
        self.assertEqual(result.intent, QueryIntent.trip_planning)


if __name__ == "__main__":
    unittest.main()
