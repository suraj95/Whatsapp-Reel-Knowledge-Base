"""Overpass API client for OSM POI queries."""

from __future__ import annotations

from typing import Any, Dict, List

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def overpass_query(query: str, *, timeout: float = 45.0) -> Dict[str, Any]:
    resp = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=timeout,
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
    )
    resp.raise_for_status()
    return resp.json()


def tourism_food_parks_around(
    lat: float,
    lon: float,
    *,
    radius_m: int = 2500,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Tourism attractions, restaurants, parks near a point. Returns simplified elements.
    """
    lim = max(5, min(limit, 40))
    q = f"""
[out:json][timeout:25];
(
  node["tourism"](around:{radius_m},{lat},{lon});
  node["amenity"="restaurant"](around:{radius_m},{lat},{lon});
  way["leisure"="park"](around:{radius_m},{lat},{lon});
);
out center;
"""
    try:
        data = overpass_query(q.strip())
    except Exception:
        return []
    elements = data.get("elements") or []
    out: List[Dict[str, Any]] = []
    for el in elements[:lim]:
        tags = el.get("tags") or {}
        name = tags.get("name") or tags.get("operator")
        if not name:
            continue
        lat_e = el.get("lat")
        lon_e = el.get("lon")
        if lat_e is None and el.get("center"):
            lat_e = el["center"].get("lat")
            lon_e = el["center"].get("lon")
        kind = tags.get("tourism") or tags.get("amenity") or tags.get("leisure") or "place"
        out.append(
            {
                "name": str(name)[:120],
                "kind": str(kind),
                "lat": lat_e,
                "lon": lon_e,
            }
        )
    return out


def railway_bus_stops_around(
    lat: float,
    lon: float,
    *,
    radius_m: int = 2000,
    limit: int = 15,
) -> List[Dict[str, Any]]:
    lim = max(3, min(limit, 30))
    q = f"""
[out:json][timeout:25];
(
  node["railway"="station"](around:{radius_m},{lat},{lon});
  node["railway"="halt"](around:{radius_m},{lat},{lon});
  node["public_transport"="station"](around:{radius_m},{lat},{lon});
  node["highway"="bus_stop"](around:{radius_m},{lat},{lon});
);
out body;
"""
    try:
        data = overpass_query(q.strip())
    except Exception:
        return []
    elements = data.get("elements") or []
    out: List[Dict[str, Any]] = []
    for el in elements[:lim]:
        tags = el.get("tags") or {}
        name = tags.get("name")
        if not name:
            continue
        lat_e = el.get("lat")
        lon_e = el.get("lon")
        if lat_e is None and el.get("center"):
            lat_e = el["center"].get("lat")
            lon_e = el["center"].get("lon")
        railway = tags.get("railway")
        hw = tags.get("highway")
        kind = "station" if railway else ("bus_stop" if hw == "bus_stop" else "transit")
        out.append({"name": str(name)[:120], "kind": kind, "lat": lat_e, "lon": lon_e})
    return out
