"""Open-Meteo forecast API — no API key."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
from .cache import get_json, make_key, set_json


def fetch_forecast_daily(
    lat: float,
    lon: float,
    *,
    days: int = 7,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """
    Return Open-Meteo daily forecast JSON (time, weathercode, temperature_2m_max, etc.).
    """
    d = max(1, min(int(days), 16))
    ttl_sec = int(os.getenv("CACHE_TTL_OPEN_METEO_SEC", "3600"))
    lat_r = round(float(lat), 3)
    lon_r = round(float(lon), 3)
    cache_key = make_key("open_meteo_daily_v1", [str(lat_r), str(lon_r), str(d)])
    cached = get_json(cache_key)
    if isinstance(cached, dict):
        return cached
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": d,
            "timezone": "auto",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    out = resp.json()
    set_json(cache_key, out, ttl_sec)
    return out


def summarize_forecast_for_prompt(forecast: Dict[str, Any], max_days: int = 5) -> List[Dict[str, Any]]:
    """Compact list of {date, max_c, min_c, precip_pct, code} for LLM context."""
    daily = forecast.get("daily") or {}
    times: List[str] = list(daily.get("time") or [])
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    pprob = daily.get("precipitation_probability_max") or []
    codes = daily.get("weathercode") or []
    out: List[Dict[str, Any]] = []
    for i, day in enumerate(times[:max_days]):
        row: Dict[str, Any] = {"date": day}
        if i < len(tmax):
            row["max_c"] = tmax[i]
        if i < len(tmin):
            row["min_c"] = tmin[i]
        if i < len(pprob):
            row["precip_pct"] = pprob[i]
        if i < len(codes):
            row["weathercode"] = codes[i]
        out.append(row)
    return out
