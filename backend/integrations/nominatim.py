"""
Nominatim (OpenStreetMap) search — no API key; respect usage policy (User-Agent, rate limits).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from .cache import get_json, make_key, set_json

DEFAULT_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "TravelReelsKB/1.0 (https://github.com/local/travel-reels-kb)",
)

_last_request_monotonic: float = 0.0
_MIN_INTERVAL_SEC = 1.05
_TTL_SEC = int(os.getenv("CACHE_TTL_NOMINATIM_SEC", "604800"))


def _throttle() -> None:
    global _last_request_monotonic
    now = time.monotonic()
    elapsed = now - _last_request_monotonic
    if elapsed < _MIN_INTERVAL_SEC:
        time.sleep(_MIN_INTERVAL_SEC - elapsed)
    _last_request_monotonic = time.monotonic()


def nominatim_search(
    query: str,
    *,
    limit: int = 1,
    timeout: float = 12.0,
    user_agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    cache_key = make_key("nominatim_search_v1", [q, str(max(1, min(limit, 10)))])
    cached = get_json(cache_key)
    if isinstance(cached, list):
        return cached

    _throttle()
    headers = {"User-Agent": user_agent or DEFAULT_USER_AGENT}
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": q,
            "format": "json",
            "limit": max(1, min(limit, 10)),
            "addressdetails": 1,
        },
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    out = data if isinstance(data, list) else []
    set_json(cache_key, out, _TTL_SEC)
    return out


def nominatim_to_lat_lon_bbox(
    query: str,
    *,
    timeout: float = 12.0,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[Tuple[float, float, float, float]]]:
    results = nominatim_search(query, limit=1, timeout=timeout)
    if not results:
        return None, None, None, None
    top = results[0]
    try:
        lat = float(top.get("lat"))
        lon = float(top.get("lon"))
    except (TypeError, ValueError):
        return None, None, None, None
    name = top.get("display_name") or query
    bbox_raw = top.get("boundingbox")
    bbox: Optional[Tuple[float, float, float, float]] = None
    if isinstance(bbox_raw, list) and len(bbox_raw) >= 4:
        try:
            south = float(bbox_raw[0])
            north = float(bbox_raw[1])
            west = float(bbox_raw[2])
            east = float(bbox_raw[3])
            bbox = (south, north, west, east)
        except (TypeError, ValueError):
            bbox = None
    return lat, lon, name, bbox


def bbox_around_point(lat: float, lon: float, radius_km: float = 3.0) -> Tuple[float, float, float, float]:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.2, abs(math.cos(math.radians(lat)))))
    return lat - dlat, lat + dlon, lon - dlon, lon + dlon
