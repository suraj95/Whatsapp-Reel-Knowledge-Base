"""Shared trip day bucketing helpers."""

from __future__ import annotations

import re
from typing import Dict, List, TypeVar

T = TypeVar("T")


def extract_trip_days(query: str) -> int:
    lowered = (query or "").lower()
    m = re.search(r"(\d+)\s*day", lowered)
    if not m:
        return 2
    try:
        value = int(m.group(1))
        return max(1, min(7, value))
    except Exception:
        return 2


def build_day_buckets(sources: List[T], days: int) -> Dict[int, List[T]]:
    buckets: Dict[int, List[T]] = {day: [] for day in range(1, days + 1)}
    if not sources:
        return buckets
    i = 0
    for source in sources:
        day = (i % days) + 1
        buckets[day].append(source)
        i += 1
    return buckets
