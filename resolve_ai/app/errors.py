"""Consistent error schema (FR-021, AC-11, NFR-009)."""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

log = logging.getLogger("resolve.errors")


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details=None):
        self.status_code, self.code, self.message, self.details = status_code, code, message, details


def not_found(what: str = "Resource") -> AppError:
    return AppError(404, "not_found", f"{what} not found.")


def _body(request: Request, code: str, message: str, details=None) -> dict:
    err = {"code": code, "message": message, "request_id": getattr(request.state, "request_id", None)}
    if details:
        err["details"] = details
    return {"error": err}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return JSONResponse(_body(request, exc.code, exc.message, exc.details), exc.status_code)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
                429: "rate_limited"}.get(exc.status_code, "http_error")
        resp = JSONResponse(_body(request, code, str(exc.detail)), exc.status_code)
        for k, v in (exc.headers or {}).items():
            resp.headers[k] = v
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = [
            {"field": ".".join(str(p) for p in e["loc"] if p not in ("body", "query", "path")),
             "issue": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse(_body(request, "validation_error", "The request was invalid.", details), 422)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error", extra={"request_id": getattr(request.state, "request_id", None)})
        return JSONResponse(
            _body(request, "internal_error", "Something went wrong on our side. Please try again later."), 500
        )
