import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import Tenant, User
from app.security import hash_password

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
PASSWORD = "correct-horse-battery"


# Run the whole suite against real PostgreSQL + pgvector with:
#   TEST_DATABASE_URL=postgresql+psycopg://user:pw@localhost:5432/resolve_test pytest
TEST_DB = os.environ.get("TEST_DATABASE_URL")


def make_settings(**kw) -> Settings:
    base = dict(database_url=TEST_DB or "sqlite://", jwt_secret="test-secret-for-tests-only-0123456789",
                rate_limit_enabled=False, log_level="WARNING")
    base.update(kw)
    return Settings(**base)


def _reset_postgres(url: str) -> None:
    from sqlalchemy import text
    from app.db import Base, make_engine
    import app.models  # noqa: F401
    eng = make_engine(url)
    with eng.begin() as c:
        c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    eng.dispose()


@pytest.fixture
def app():
    if TEST_DB:
        _reset_postgres(TEST_DB)  # clean slate per test
    return create_app(make_settings())


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


class Env:
    """Convenience helpers around a running app."""

    def __init__(self, app, client):
        self.app, self.client = app, client

    def tenant(self, slug: str, role: str = "admin", email: str | None = None) -> dict:
        sf = self.app.state.session_factory
        with sf() as db:
            t = db.query(Tenant).filter_by(slug=slug).first() or Tenant(name=slug.title(), slug=slug)
            db.add(t); db.flush()
            email = email or f"{role}@{slug}.test"
            db.add(User(tenant_id=t.id, email=email, password_hash=hash_password(PASSWORD), role=role))
            db.commit()
            tid = t.id
        r = self.client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
        assert r.status_code == 200, r.text
        return {"id": tid, "slug": slug, "headers": {"Authorization": f"Bearer {r.json()['access_token']}"}}

    def upload(self, headers, name: str, data: bytes, ctype: str = "text/markdown", title: str | None = None):
        return self.client.post("/api/v1/sources", headers=headers, files={"file": (name, io.BytesIO(data), ctype)},
                                data={"title": title} if title else None)

    def load_samples(self, headers) -> None:
        titles = {"return_policy.md": "Return Policy", "shipping_faq.md": "Shipping and Delivery FAQ",
                  "product_faq.md": "Product FAQ"}
        for p in sorted(SAMPLE.glob("*.md")):
            r = self.upload(headers, p.name, p.read_bytes(), title=titles[p.name])
            assert r.status_code == 202, r.text

    def conversation(self, slug: str) -> dict:
        r = self.client.post("/api/v1/conversations", json={"tenant": slug})
        assert r.status_code == 201, r.text
        d = r.json()
        d["headers"] = {"Authorization": f"Bearer {d['access_token']}"}
        return d

    def say(self, conv: dict, text: str) -> dict:
        r = self.client.post(f"/api/v1/conversations/{conv['id']}/messages", headers=conv["headers"],
                             json={"message": text})
        assert r.status_code == 200, r.text
        return r.json()


@pytest.fixture
def env(app, client):
    return Env(app, client)


@pytest.fixture
def store(env):
    """Tenant 'demo-store' with sample knowledge loaded."""
    t = env.tenant("demo-store")
    env.load_samples(t["headers"])
    return t
