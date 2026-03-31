import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


logger = logging.getLogger(__name__)
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_MEMORY_CACHE = {}
_DEFAULT_PREFIX = "travelkb:cache"


def _client():
    if redis is None:
        return None
    try:
        c = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
        c.ping()
        return c
    except Exception:
        return None


def key_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def make_key(namespace: str, parts: list[str]) -> str:
    normalized = [str(p).strip().lower() for p in parts]
    return f"{_DEFAULT_PREFIX}:{namespace}:{':'.join(normalized)}"


def get_json(key: str) -> Optional[Any]:
    c = _client()
    if c is None:
        row = _MEMORY_CACHE.get(key)
        if not row or row["expires_at"] < time.time():
            return None
        logger.info("cache_hit backend=memory key=%s", key)
        return row["value"]
    raw = c.get(key)
    if not raw:
        return None
    logger.info("cache_hit backend=redis key=%s", key)
    return json.loads(raw)


def set_json(key: str, value: Any, ttl_sec: int) -> None:
    c = _client()
    if c is None:
        _MEMORY_CACHE[key] = {"value": value, "expires_at": time.time() + ttl_sec}
        return
    c.setex(key, max(1, int(ttl_sec)), json.dumps(value))

