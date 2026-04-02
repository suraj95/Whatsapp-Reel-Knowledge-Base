import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..observability import (
    clear_request_context,
    get_log,
    get_request_id,
    normalize_request_id,
    reset_request_timings,
    set_request_id,
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = normalize_request_id(request.headers.get("x-request-id"))
        set_request_id(rid)
        reset_request_timings()
        t0 = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception as ex:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            get_log().exception(
                "http_request_failed",
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=round(elapsed_ms, 2),
                error_type=type(ex).__name__,
            )
            clear_request_context()
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = get_request_id() or rid
        get_log().info(
            "http_request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(elapsed_ms, 2),
        )
        clear_request_context()
        return response
