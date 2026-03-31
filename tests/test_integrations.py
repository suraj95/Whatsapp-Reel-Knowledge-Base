"""Mocked HTTP tests for integration clients."""

import unittest
from unittest.mock import MagicMock, patch

from backend.integrations import open_meteo
from backend.integrations.nominatim import nominatim_search
from backend.integrations.overpass import overpass_query
from backend.trip_utils import build_day_buckets, extract_trip_days


class TripUtilsTests(unittest.TestCase):
    def test_extract_trip_days(self):
        self.assertEqual(extract_trip_days("3 day trip to Paris"), 3)
        self.assertEqual(extract_trip_days("weekend"), 2)

    def test_buckets(self):
        items = ["a", "b", "c"]
        b = build_day_buckets(items, 2)
        self.assertEqual(len(b[1]) + len(b[2]), 3)


class NominatimTests(unittest.TestCase):
    @patch("backend.integrations.nominatim.requests.get")
    @patch("backend.integrations.nominatim.set_json")
    @patch("backend.integrations.nominatim.get_json")
    def test_search_parses(self, mock_cache_get, mock_cache_set, mock_get):
        mock_cache_get.return_value = None
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "lat": "48.85",
                "lon": "2.35",
                "display_name": "Paris, France",
                "boundingbox": ["48.8", "48.9", "2.2", "2.4"],
                "address": {"city": "Paris", "country": "France"},
            }
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        out = nominatim_search("Paris", limit=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["lat"], "48.85")
        mock_cache_set.assert_called_once()

    @patch("backend.integrations.nominatim.requests.get")
    @patch("backend.integrations.nominatim.get_json")
    def test_search_cache_hit_skips_http(self, mock_cache_get, mock_get):
        mock_cache_get.return_value = [{"lat": "48.85", "lon": "2.35"}]
        out = nominatim_search("Paris", limit=1)
        self.assertEqual(len(out), 1)
        mock_get.assert_not_called()


class OpenMeteoTests(unittest.TestCase):
    @patch("backend.integrations.open_meteo.requests.get")
    @patch("backend.integrations.open_meteo.set_json")
    @patch("backend.integrations.open_meteo.get_json")
    def test_forecast(self, mock_cache_get, mock_cache_set, mock_get):
        mock_cache_get.return_value = None
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "daily": {
                "time": ["2026-01-01"],
                "temperature_2m_max": [10],
                "temperature_2m_min": [5],
                "precipitation_probability_max": [20],
                "weathercode": [0],
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        raw = open_meteo.fetch_forecast_daily(0.0, 0.0, days=1)
        summ = open_meteo.summarize_forecast_for_prompt(raw, max_days=1)
        self.assertEqual(len(summ), 1)
        self.assertEqual(summ[0]["date"], "2026-01-01")
        mock_cache_set.assert_called_once()

    @patch("backend.integrations.open_meteo.requests.get")
    @patch("backend.integrations.open_meteo.get_json")
    def test_forecast_cache_hit_skips_http(self, mock_cache_get, mock_get):
        mock_cache_get.return_value = {"daily": {"time": ["2026-01-01"]}}
        raw = open_meteo.fetch_forecast_daily(0.0, 0.0, days=1)
        self.assertIn("daily", raw)
        mock_get.assert_not_called()


class OverpassTests(unittest.TestCase):
    @patch("backend.integrations.overpass.requests.post")
    @patch("backend.integrations.overpass.set_json")
    @patch("backend.integrations.overpass.get_json")
    def test_overpass_query_cache_behavior(self, mock_cache_get, mock_cache_set, mock_post):
        mock_cache_get.return_value = None
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"elements": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        out = overpass_query("[out:json];node(1,2,3,4);out;")
        self.assertEqual(out["elements"], [])
        mock_cache_set.assert_called_once()

    @patch("backend.integrations.overpass.requests.post")
    @patch("backend.integrations.overpass.get_json")
    def test_overpass_query_cache_hit_skips_http(self, mock_cache_get, mock_post):
        mock_cache_get.return_value = {"elements": [{"id": 1}]}
        out = overpass_query("query")
        self.assertEqual(len(out["elements"]), 1)
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
