"""AC-01, AC-02, AC-10, FR-003/004/008/021."""
import io

from tests.conftest import PASSWORD


def test_ac01_login_and_protected_routes(env, client):
    assert client.get("/api/v1/sources").status_code == 401           # unauthenticated rejected
    assert client.get("/api/v1/sources", headers={"Authorization": "Bearer junk"}).status_code == 401
    t = env.tenant("acme")
    assert client.get("/api/v1/sources", headers=t["headers"]).status_code == 200
    bad = client.post("/api/v1/auth/login", json={"email": "admin@acme.test", "password": "wrong-password"})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "unauthorized"
    unknown = client.post("/api/v1/auth/login", json={"email": "nobody@acme.test", "password": PASSWORD})
    assert unknown.status_code == 401 and unknown.json()["error"]["message"] == bad.json()["error"]["message"]


def test_operator_cannot_manage_sources(env, client):
    env.tenant("acme")
    op = env.tenant("acme", role="operator", email="op@acme.test")
    assert client.get("/api/v1/sources", headers=op["headers"]).status_code == 403


def test_customer_token_cannot_use_admin_routes(env, client):
    env.tenant("acme")
    conv = env.conversation("acme")
    assert client.get("/api/v1/sources", headers=conv["headers"]).status_code == 403


def test_ac02_upload_shows_status_and_chunks(env, client):
    t = env.tenant("acme")
    r = env.upload(t["headers"], "faq.txt", b"# Returns\nItems can be returned within 30 days.", "text/plain", "Returns FAQ")
    assert r.status_code == 202 and r.json()["title"] == "Returns FAQ"
    sid = r.json()["id"]
    got = client.get(f"/api/v1/sources/{sid}", headers=t["headers"]).json()
    assert got["status"] == "ready" and got["chunk_count"] >= 1   # background task finished
    listing = client.get("/api/v1/sources", headers=t["headers"]).json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == sid


def test_upload_validation_errors_are_actionable(env, client, app):
    t = env.tenant("acme")
    h = t["headers"]
    cases = [
        (("a.exe", b"MZ", "application/octet-stream"), 415, "Unsupported file type"),
        (("a.txt", b"", "text/plain"), 422, "empty"),
        (("a.txt", b"   \n ", "text/plain"), 422, "no text"),
        (("a.txt", b"hello\x00world", "text/plain"), 422, "binary"),
        (("a.txt", "héllo".encode("latin-1"), "text/plain"), 422, "UTF-8"),
        (("a.pdf", b"not a pdf at all", "application/pdf"), 422, "not a valid PDF"),
        (("a.pdf", b"%PDF-1.4 garbage", "application/pdf"), 422, "could not be read"),
        (("a.txt", b"hello", "image/png"), 415, "content type"),
    ]
    for (name, data, ctype), status, needle in cases:
        r = env.upload(h, name, data, ctype)
        assert r.status_code == status, (name, r.text)
        assert needle.lower() in r.json()["error"]["message"].lower()
    assert client.get("/api/v1/sources", headers=h).json()["total"] == 0


def test_oversized_upload_rejected():
    from fastapi.testclient import TestClient
    from app.main import create_app
    from tests.conftest import Env, make_settings
    from tests.conftest import TEST_DB, _reset_postgres
    TEST_DB and _reset_postgres(TEST_DB)
    app = create_app(make_settings(max_upload_bytes=100))
    with TestClient(app) as c:
        e = Env(app, c)
        t = e.tenant("acme")
        r = e.upload(t["headers"], "big.txt", b"x " * 500, "text/plain")
        assert r.status_code == 413 and "too large" in r.json()["error"]["message"]


def test_duplicate_upload_conflict(env, client):
    t = env.tenant("acme")
    assert env.upload(t["headers"], "a.txt", b"Same content here.", "text/plain").status_code == 202
    r = env.upload(t["headers"], "b.txt", b"Same content here.", "text/plain")
    assert r.status_code == 409 and r.json()["error"]["code"] == "duplicate_source"


def test_pdf_upload_extracts_pages_with_citations(env, client):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 750, "Cover page. Welcome to the Acme handbook.")
    c.showPage()
    c.drawString(72, 750, "Gift wrapping is available for 50 rupees per item.")
    c.showPage()
    c.save()
    t = env.tenant("acme")
    r = env.upload(t["headers"], "handbook.pdf", buf.getvalue(), "application/pdf", "Handbook")
    assert r.status_code == 202
    assert client.get(f"/api/v1/sources/{r.json()['id']}", headers=t["headers"]).json()["status"] == "ready"
    conv = env.conversation("acme")
    reply = env.say(conv, "How much does gift wrapping cost per item?")
    assert reply["status"] == "answered"
    assert {"source_id": r.json()["id"], "title": "Handbook", "page": 2, "section": None} in reply["sources"]


def test_ac10_delete_source_removes_it_from_retrieval(env, client, app):
    t = env.tenant("acme")
    sid = env.upload(t["headers"], "w.txt", b"# Gift wrapping\nGift wrapping costs 50 rupees per item.",
                     "text/plain", "Wrapping").json()["id"]
    conv = env.conversation("acme")
    assert env.say(conv, "How much does gift wrapping cost?")["status"] == "answered"
    assert client.delete(f"/api/v1/sources/{sid}", headers=t["headers"]).status_code == 204
    again = env.say(conv, "How much does gift wrapping cost?")
    assert again["status"] == "fallback" and again["sources"] == []
    from app.models import DocumentChunk
    with app.state.session_factory() as db:
        assert db.query(DocumentChunk).count() == 0               # vectors really gone
    assert client.get(f"/api/v1/sources/{sid}", headers=t["headers"]).status_code == 404


def test_failed_embedding_marks_source_failed_and_indexes_nothing(app, client):
    from app.providers import ProviderError
    from tests.conftest import Env

    class Boom:
        dim = 1024
        recommended_min_score = 0.2
        def embed(self, texts): raise ProviderError("down")

    app.state.embedder = Boom()
    env = Env(app, client)
    t = env.tenant("acme")
    r = env.upload(t["headers"], "a.txt", b"# T\nSome policy text.", "text/plain")
    src = client.get(f"/api/v1/sources/{r.json()['id']}", headers=t["headers"]).json()
    assert src["status"] == "failed" and "embedding" in src["error"].lower() and src["chunk_count"] == 0
    from app.models import DocumentChunk
    with app.state.session_factory() as db:
        assert db.query(DocumentChunk).count() == 0
