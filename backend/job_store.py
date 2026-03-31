import json
import logging
import os
import time
from typing import Any, Dict, Optional

try:
    import redis
except Exception:  # pragma: no cover - optional dependency in some envs
    redis = None


logger = logging.getLogger(__name__)

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_STATUS_TTL_SEC = int(os.getenv("INGESTION_STATUS_TTL_SEC", "86400"))
_in_memory_jobs: Dict[str, Dict[str, Any]] = {}


def _get_redis_client():
    if redis is None:
        return None
    try:
        client = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception as ex:
        logger.warning("redis_unavailable fallback=in_memory reason=%s", ex)
        return None


def _job_key(job_id: str) -> str:
    return f"ingestion:job:{job_id}"


def set_job_status(job_id: str, payload: Dict[str, Any]) -> None:
    data = dict(payload)
    data["updated_at"] = int(time.time())
    client = _get_redis_client()
    if client is None:
        _in_memory_jobs[job_id] = data
        return
    client.setex(_job_key(job_id), _STATUS_TTL_SEC, json.dumps(data))


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    client = _get_redis_client()
    if client is None:
        return _in_memory_jobs.get(job_id)
    raw = client.get(_job_key(job_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

