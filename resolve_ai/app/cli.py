"""Operational CLI.

    python -m app.cli create-tenant --name "Acme" --email admin@acme.com --password '...'
    python -m app.cli seed-demo
    python -m app.cli purge-conversations
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, select

from app.config import get_settings
from app.db import make_engine, make_session_factory
from app.models import Conversation, KnowledgeSource, Tenant, User
from app.providers import build_providers
from app.security import hash_password, slugify
from app.services.ingestion import checksum, index_source, validate_and_extract

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
SAMPLE_TITLES = {"return_policy.md": "Return Policy", "shipping_faq.md": "Shipping and Delivery FAQ",
                 "product_faq.md": "Product FAQ"}


def _session():
    s = get_settings()
    engine = make_engine(s.database_url)
    if s.database_url.startswith("sqlite"):  # dev convenience; PostgreSQL schemas come from Alembic
        from app.db import Base
        Base.metadata.create_all(engine)
    return s, make_session_factory(engine)()


def create_tenant(name: str, email: str, password: str, slug: str | None = None, role: str = "admin") -> Tenant:
    if len(password) < 8:
        raise SystemExit("Password must be at least 8 characters.")
    _, db = _session()
    slug = slug or slugify(name)
    tenant = db.scalar(select(Tenant).where(Tenant.slug == slug))
    if tenant is None:
        tenant = Tenant(name=name, slug=slug)
        db.add(tenant)
        db.flush()
    if db.scalar(select(User).where(User.email == email.lower())):
        raise SystemExit(f"User {email} already exists.")
    db.add(User(tenant_id=tenant.id, email=email.lower(), password_hash=hash_password(password), role=role))
    db.commit()
    print(f"Tenant '{tenant.name}' (slug: {tenant.slug}) ready; {role} user {email} created.")
    return tenant


def seed_demo(password: str) -> None:
    settings, db = _session()
    _, embedder = build_providers(settings)
    tenant = db.scalar(select(Tenant).where(Tenant.slug == "demo-store"))
    if tenant is None:
        tenant = create_tenant("Demo Store", "admin@demo-store.test", password, slug="demo-store")
        create_tenant("Demo Store", "operator@demo-store.test", password, slug="demo-store", role="operator")
        _, db = _session()
    for path in sorted(SAMPLE_DIR.glob("*.md")):
        data = path.read_bytes()
        if db.scalar(select(KnowledgeSource).where(KnowledgeSource.tenant_id == tenant.id,
                                                   KnowledgeSource.checksum == checksum(data))):
            continue
        stype, pages = validate_and_extract(path.name, "text/markdown", data, settings.max_upload_bytes)
        src = KnowledgeSource(tenant_id=tenant.id, title=SAMPLE_TITLES.get(path.name, path.stem), filename=path.name,
                              source_type=stype, status="processing", checksum=checksum(data))
        db.add(src)
        db.commit()
        index_source(make_session_factory(db.get_bind()), embedder, src.id, tenant.id, pages,
                     settings.chunk_size, settings.chunk_overlap)
        print(f"Indexed {path.name}")
    print(f"\nDemo ready. Open http://localhost:8000  (tenant slug: demo-store)")
    print("Admin login: admin@demo-store.test  /  the password you provided")
    print("Try orders: ORD-1001 + alice@example.com | ORD-1002 + bob@example.com | ORD-9999 (simulated outage)")


def purge_conversations() -> None:
    s, db = _session()
    if s.conversation_retention_days <= 0:
        print("CONVERSATION_RETENTION_DAYS is 0 (keep forever); nothing to do.")
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=s.conversation_retention_days)
    res = db.execute(delete(Conversation).where(Conversation.updated_at < cutoff))
    db.commit()
    print(f"Deleted {res.rowcount} conversation(s) older than {s.conversation_retention_days} days "
          "(messages, tickets and tool records are removed with them).")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="resolve-ai")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("create-tenant")
    t.add_argument("--name", required=True); t.add_argument("--email", required=True)
    t.add_argument("--password", required=True); t.add_argument("--slug")
    t.add_argument("--role", choices=["admin", "operator"], default="admin")
    d = sub.add_parser("seed-demo"); d.add_argument("--password", default="demo-password-123")
    sub.add_parser("purge-conversations")
    a = p.parse_args(argv)
    if a.cmd == "create-tenant":
        create_tenant(a.name, a.email, a.password, a.slug, a.role)
    elif a.cmd == "seed-demo":
        seed_demo(a.password)
    else:
        purge_conversations()


if __name__ == "__main__":
    main(sys.argv[1:])
