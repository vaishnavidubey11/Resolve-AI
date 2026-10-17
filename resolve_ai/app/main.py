from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import auth, conversations, health, sources, tickets
from app.config import Settings, get_settings
from app.db import Base, make_engine, make_session_factory
from app.errors import register_error_handlers
from app.logging_config import configure_logging
from app.middleware import RequestContextMiddleware, SlidingWindowLimiter
from app.providers import build_providers
from app.services.orders import OrderClient, build_order_client

STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, *, llm=None, embedder=None, order_client: OrderClient | None = None,
               ) -> FastAPI:
    """Application factory. Providers and the order client can be injected (used heavily in tests)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    if settings.jwt_secret == Settings.model_fields["jwt_secret"].default and not settings.database_url.startswith("sqlite"):
        raise RuntimeError("Refusing to start: set JWT_SECRET to a long random value (see .env.example).")
    engine = make_engine(settings.database_url)
    if settings.database_url.startswith("sqlite"):
        import app.models  # noqa: F401  (register tables)
        Base.metadata.create_all(engine)  # dev convenience only; PostgreSQL uses Alembic migrations
    default_llm, default_emb = (None, None)
    if llm is None or embedder is None:
        default_llm, default_emb = build_providers(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        engine.dispose()

    app = FastAPI(
        title="Resolve AI", version=__version__, lifespan=lifespan,
        description="AI customer support automation: grounded RAG answers, order lookup and human escalation.",
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.llm = llm or default_llm
    app.state.embedder = embedder or default_emb
    app.state.order_client = order_client or build_order_client(settings)
    app.state.limiter = SlidingWindowLimiter()

    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(health.router)
    for r in (auth.router, sources.router, conversations.router, tickets.router):
        app.include_router(r, prefix="/api/v1")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def _default_app() -> FastAPI:
    return create_app()


# `uvicorn app.main:app` - created lazily so importing this module has no side effects in tests.
def __getattr__(name: str):
    if name == "app":
        globals()["app"] = _default_app()
        return globals()["app"]
    raise AttributeError(name)
