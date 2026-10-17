"""Pure unit tests: chunking, providers, security helpers, retention (no HTTP)."""
import pytest

from app.config import Settings
from app.providers import ProviderError
from app.providers.mock import MockEmbeddingProvider, MockLLMProvider
from app.providers.openai_compat import OpenAIEmbeddingProvider, OpenAILLMProvider
from app.security import create_token, decode_token, hash_password, verify_password
from app.services.ingestion import chunk_pages
from app.services.retrieval import _cosine


def test_chunking_keeps_sections_pages_and_respects_size():
    text = "# Returns\n" + " ".join(f"Sentence number {i} about returns." for i in range(60)) + "\n\n# Shipping\nShips fast."
    chunks = chunk_pages([(3, text)], size=300, overlap=60)
    assert all(len(c.text) <= 300 + len("Returns\n") + 5 for c in chunks)
    assert {c.section for c in chunks} == {"Returns", "Shipping"} and {c.page for c in chunks} == {3}
    assert [c.index for c in chunks] == list(range(len(chunks)))
    returns = [c for c in chunks if c.section == "Returns"]
    assert len(returns) > 2
    # overlap: consecutive chunks share at least one sentence
    norm = lambda t: {x.strip(".") for x in t.split("\n", 1)[1].split(". ")}
    assert norm(returns[0].text) & norm(returns[1].text)
    assert chunks[-1].text == "Shipping\nShips fast."


def test_chunking_handles_long_unbroken_text_and_plain_headings():
    long = "word" * 2000
    assert all(len(c.text) <= 600 for c in chunk_pages([(None, long)], size=500, overlap=50))
    c = chunk_pages([(None, "Refund Policy\nRefunds take 5 days.\n\nShipping Times\nShipping takes 3 days.")], 500, 50)
    assert [x.section for x in c] == ["Refund Policy", "Shipping Times"]
    assert chunk_pages([(None, "   ")], 500, 50) == []


def test_mock_embeddings_are_deterministic_and_normalized():
    e = MockEmbeddingProvider(64)
    a, b = e.embed(["return policy"])[0], e.embed(["return policy"])[0]
    assert a == b and abs(sum(x * x for x in a) - 1) < 1e-9
    assert _cosine(a, e.embed(["return policy items"])[0]) > _cosine(a, e.embed(["banana smoothie"])[0])


def test_mock_llm_only_uses_supplied_context():
    from app.providers import ContextPassage
    p = ContextPassage("s", "T", None, "Refunds", "Refunds\nRefunds take 5 days. Shipping is free.", 0.9)
    out = MockLLMProvider().generate("sys", [], "how long do refunds take", [p])
    assert "5 days" in out and "Refunds\n" not in out
    assert MockLLMProvider().generate("sys", [], "q", []) .startswith("I don't have enough")


def test_password_hashing_and_tokens():
    h = hash_password("s3cret-pass")
    assert h != "s3cret-pass" and verify_password("s3cret-pass", h) and not verify_password("nope", h)
    assert not verify_password("x", "garbage")
    s = Settings(jwt_secret="k" * 32)
    tok = create_token(s, {"typ": "staff", "sub": "u1"}, 5)
    assert decode_token(s, tok)["sub"] == "u1"
    assert decode_token(Settings(jwt_secret="other" * 8), tok) is None
    assert decode_token(s, create_token(s, {"sub": "u"}, -1)) is None          # expired


def test_openai_provider_requires_key_and_surfaces_errors(monkeypatch):
    with pytest.raises(ProviderError):
        OpenAILLMProvider(Settings(openai_api_key=""))
    s = Settings(openai_api_key="sk-test", openai_base_url="http://127.0.0.1:9", provider_max_retries=0,
                 provider_timeout_seconds=0.2)
    with pytest.raises(ProviderError) as e:
        OpenAIEmbeddingProvider(s).embed(["hi"])
    assert "sk-test" not in str(e.value)


def test_openai_provider_parses_responses_and_retries(monkeypatch):
    import httpx
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.url.path.endswith("/embeddings"):
            if calls["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": " Hello "}}]})

    s = Settings(openai_api_key="sk-test", openai_base_url="http://x/v1", provider_max_retries=2, embedding_dim=2)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    emb, llm = OpenAIEmbeddingProvider(s), OpenAILLMProvider(s)
    for p in (emb, llm):
        p.http.client = httpx.Client(base_url="http://x/v1", transport=httpx.MockTransport(handler))
    assert emb.embed(["a", "b"]) == [[1, 0], [0, 1]] and calls["n"] == 2     # retried once, order restored
    assert llm.generate("sys", [], "q", []) == "Hello"


def test_retention_purge(monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone
    from app import cli
    from app.db import Base, make_engine, make_session_factory
    from app.models import Conversation, Message, Tenant
    url = f"sqlite:///{tmp_path}/t.db"
    eng = make_engine(url); Base.metadata.create_all(eng)
    sf = make_session_factory(eng)
    with sf() as db:
        t = Tenant(name="T", slug="t"); db.add(t); db.flush()
        old = Conversation(tenant_id=t.id, updated_at=datetime.now(timezone.utc) - timedelta(days=40))
        new = Conversation(tenant_id=t.id)
        db.add_all([old, new]); db.flush()
        db.add(Message(conversation_id=old.id, seq=1, role="customer", content="hi")); db.commit()
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(database_url=url, conversation_retention_days=30))
    cli.purge_conversations()
    with sf() as db:
        assert db.query(Conversation).count() == 1 and db.query(Message).count() == 0


def test_refuses_default_jwt_secret_outside_dev():
    from app.main import create_app
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        create_app(Settings(database_url="postgresql+psycopg://u:p@localhost/db"))
