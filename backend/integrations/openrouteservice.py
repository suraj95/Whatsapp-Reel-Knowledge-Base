"""Optional OpenRouteService directions (free tier with API key)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


def directions_geojson(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    *,
    profile: str = "driving-car",
    timeout: float = 20.0,
) -> Optional[Dict[str, Any]]:
    key = os.getenv("OPENROUTESERVICE_API_KEY")
    if not key:
        return None
    resp = requests.post(
        "https://api.openrouteservice.org/v2/directions/" + profile,
        json={
            "coordinates": [[start_lon, start_lat], [end_lon, end_lat]],
            "instructions": False,
        },
        headers={"Authorization": key, "Content-Type": "application/json"},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        return None
    return resp.json()


def summarize_route_distance_km(geojson: Optional[Dict[str, Any]]) -> Optional[float]:
    if not geojson:
        return None
    try:
        routes = geojson.get("routes") or []
        if not routes:
            return None
        m = routes[0].get("summary", {}).get("distance")
        if m is None:
            return None
        return round(float(m) / 1000.0, 1)
    except (TypeError, ValueError):
        return None
