"""
Request-scoped IDs, structured logging (structlog), and step timings for observability.
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import structlog

_REQUEST_ID_RE = re.compile(r"^[a-zA-Z0-9-]{1,128}$")

_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_request_timings_var: ContextVar[Optional[Dict[str, float]]] = ContextVar("request_timings", default=None)


def configure_observability() -> None:
    """Call once at process startup (after load_dotenv)."""
    log_format = (os.getenv("LOG_FORMAT") or "console").strip().lower()
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    structlog.configure(
        processors=shared_pre_chain
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_pre_chain,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def get_log() -> structlog.stdlib.BoundLogger:
    return structlog.get_logger("app")


def normalize_request_id(header_value: Optional[str]) -> str:
    if header_value:
        s = header_value.strip()
        if _REQUEST_ID_RE.match(s):
            return s
    return str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    _request_id_var.set(None)
    _request_timings_var.set(None)
    structlog.contextvars.clear_contextvars()


def get_request_id() -> Optional[str]:
    return _request_id_var.get()


def reset_request_timings() -> None:
    _request_timings_var.set({})


def record_timing(step: str, duration_ms: float) -> None:
    d = _request_timings_var.get()
    if d is not None:
        d[step] = round(duration_ms, 2)


def get_request_timings() -> Dict[str, float]:
    d = _request_timings_var.get()
    return dict(d) if d else {}


def truncate_for_log(text: Optional[str], max_len: Optional[int] = None) -> str:
    if not text:
        return ""
    cap = max_len
    if cap is None:
        cap = 4000 if (os.getenv("LOG_QUERY_BODY") or "").strip() == "1" else 500
    s = text.strip()
    if len(s) <= cap:
        return s
    return s[:cap] + "…"


@contextmanager
def time_block(step: str, agent: str, **extra: Any) -> Iterator[None]:
    log = get_log()
    t0 = time.perf_counter()
    log.info("agent_step_start", step=step, agent=agent, **extra)
    try:
        yield
    except Exception as ex:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        record_timing(step, elapsed_ms)
        log.exception(
            "agent_step_failed",
            step=step,
            agent=agent,
            duration_ms=round(elapsed_ms, 2),
            error_type=type(ex).__name__,
            error_message=str(ex)[:500],
            **extra,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        record_timing(step, elapsed_ms)
        log.info(
            "agent_step_complete",
            step=step,
            agent=agent,
            duration_ms=round(elapsed_ms, 2),
            **extra,
        )


@asynccontextmanager
async def async_time_block(step: str, agent: str, **extra: Any) -> AsyncIterator[None]:
    log = get_log()
    t0 = time.perf_counter()
    log.info("agent_step_start", step=step, agent=agent, **extra)
    try:
        yield
    except Exception as ex:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        record_timing(step, elapsed_ms)
        log.exception(
            "agent_step_failed",
            step=step,
            agent=agent,
            duration_ms=round(elapsed_ms, 2),
            error_type=type(ex).__name__,
            error_message=str(ex)[:500],
            **extra,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        record_timing(step, elapsed_ms)
        log.info(
            "agent_step_complete",
            step=step,
            agent=agent,
            duration_ms=round(elapsed_ms, 2),
            **extra,
        )


def log_intent_resolved(
    *,
    path: str,
    intent: str,
    confidence: float,
    reason: Optional[str],
    query_preview: str,
    query_chars: int,
) -> None:
    get_log().info(
        "intent_resolved",
        agent="intent",
        path=path,
        intent=intent,
        confidence=round(confidence, 4),
        reason=reason,
        query_preview=query_preview,
        query_chars=query_chars,
    )


def log_ingestion_stage(
    *,
    event: str,
    job_id: str,
    stage: str,
    duration_ms: float,
    correlation_id: Optional[str] = None,
    **extra: Any,
) -> None:
    fields: Dict[str, Any] = {
        "job_id": job_id,
        "stage": stage,
        "duration_ms": round(duration_ms, 2),
        **extra,
    }
    if correlation_id:
        fields["correlation_id"] = correlation_id
    get_log().info(event, **fields)


def expose_observability_in_response() -> bool:
    return (os.getenv("EXPOSE_OBSERVABILITY_IN_RESPONSE") or "").strip() == "1"
