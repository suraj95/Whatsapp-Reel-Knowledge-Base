"""Open-Meteo forecast API — no API key."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


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
    return resp.json()


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
