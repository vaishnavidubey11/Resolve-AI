from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    try:
        with request.app.state.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "ok", "version": __version__}
    except Exception:
        return JSONResponse({"status": "degraded", "database": "unavailable", "version": __version__}, 503)
