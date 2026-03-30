"""Amadeus self-service API: OAuth2 + flight offers (test or prod host via env)."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests

_token_lock = threading.Lock()
_cached_token: Optional[str] = None
_token_expires_at: float = 0.0


def _base_url() -> str:
    return (os.getenv("AMADEUS_HOST") or "https://test.api.amadeus.com").rstrip("/")


def get_access_token(*, timeout: float = 15.0) -> Optional[str]:
    global _cached_token, _token_expires_at
    cid = os.getenv("AMADEUS_CLIENT_ID")
    secret = os.getenv("AMADEUS_CLIENT_SECRET")
    if not cid or not secret:
        return None

    with _token_lock:
        now = time.time()
        if _cached_token and now < _token_expires_at - 30:
            return _cached_token

        resp = requests.post(
            f"{_base_url()}/v1/security/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": cid,
                "client_secret": secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        expires_in = int(data.get("expires_in") or 1799)
        if not token:
            return None
        _cached_token = token
        _token_expires_at = now + max(60, expires_in)
        return token


def search_airport_codes(keyword: str, *, max_results: int = 3, timeout: float = 12.0) -> List[Dict[str, Any]]:
    token = get_access_token(timeout=timeout)
    if not token or not (keyword or "").strip():
        return []
    resp = requests.get(
        f"{_base_url()}/v1/reference-data/locations",
        params={
            "keyword": keyword.strip()[:80],
            "subType": "AIRPORT,CITY",
            "page[limit]": max(1, min(max_results, 10)),
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return list(data.get("data") or [])


def pick_iata_from_location_results(rows: List[Dict[str, Any]]) -> Optional[str]:
    for row in rows:
        sub = (row.get("subType") or "").upper()
        iata = row.get("iataCode")
        if iata and sub in {"AIRPORT", "CITY"}:
            return str(iata)
    return None


def flight_offers_search(
    origin_iata: str,
    dest_iata: str,
    departure_date: str,
    *,
    adults: int = 1,
    max_offers: int = 5,
    timeout: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    departure_date: YYYY-MM-DD
    Returns Amadeus flight-offer dictionaries (may be empty).
    """
    token = get_access_token(timeout=timeout)
    if not token:
        return []
    resp = requests.get(
        f"{_base_url()}/v2/shopping/flight-offers",
        params={
            "originLocationCode": origin_iata.upper()[:3],
            "destinationLocationCode": dest_iata.upper()[:3],
            "departureDate": departure_date,
            "adults": max(1, min(adults, 9)),
            "max": max(1, min(max_offers, 10)),
            "currencyCode": "USD",
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return list(data.get("data") or [])


def compact_offer_summary(offers: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    """Small JSON-safe summaries for LLM context."""
    out: List[Dict[str, Any]] = []
    for offer in offers[:limit]:
        price = (offer.get("price") or {}).get("total")
        cur = (offer.get("price") or {}).get("currency", "USD")
        its = offer.get("itineraries") or []
        segs = []
        if its:
            for leg in its[:1]:
                for s in (leg.get("segments") or [])[:2]:
                    dep = s.get("departure", {}) or {}
                    arr = s.get("arrival", {}) or {}
                    carrier = (s.get("carrierCode") or "")[:3]
                    num = str(s.get("number") or "")
                    segs.append(
                        {
                            "from": dep.get("iataCode"),
                            "to": arr.get("iataCode"),
                            "dep": dep.get("at"),
                            "arr": arr.get("at"),
                            "flight": f"{carrier}{num}".strip(),
                        }
                    )
        out.append({"price_total": price, "currency": cur, "segments": segs})
    return out
