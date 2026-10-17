import logging
import time
import uuid
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.logging_config import request_id_var

log = logging.getLogger("resolve.request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to every request and emit one structured access log line."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = rid
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        log.info(
            "request", extra={
                "request_id": rid, "method": request.method, "path": request.url.path,
                "status": response.status_code, "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response


class SlidingWindowLimiter:
    """Small in-memory limiter (single-process). Swap for Redis when scaling out."""

    def __init__(self):
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str, limit: int, window: float = 60.0) -> None:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            retry = max(1, int(window - (now - q[0])))
            raise HTTPException(429, "Too many requests. Please slow down and try again shortly.",
                                headers={"Retry-After": str(retry)})
        q.append(now)


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"
