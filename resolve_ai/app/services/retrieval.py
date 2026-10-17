"""Tenant-scoped vector retrieval (FR-011). pgvector on PostgreSQL, Python cosine elsewhere."""
import math

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import DocumentChunk, KnowledgeSource
from app.providers import ContextPassage, EmbeddingProvider


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(db: Session, embedder: EmbeddingProvider, tenant_id: str, query: str, top_k: int) -> list[ContextPassage]:
    """Top-k passages for ``query`` from READY sources owned by ``tenant_id`` only, best first."""
    qvec = embedder.embed([query])[0]
    if db.get_bind().dialect.name == "postgresql":
        rows = db.execute(
            text("""
                SELECT c.source_id, s.title, c.page, c.section, c.text,
                       1 - (c.embedding <=> CAST(:q AS vector)) AS score
                FROM document_chunks c
                JOIN knowledge_sources s ON s.id = c.source_id
                WHERE c.tenant_id = :tid AND s.tenant_id = :tid AND s.status = 'ready'
                ORDER BY c.embedding <=> CAST(:q AS vector)
                LIMIT :k
            """),
            {"q": "[" + ",".join(f"{x:.7f}" for x in qvec) + "]", "tid": tenant_id, "k": top_k},
        ).all()
        return [ContextPassage(r.source_id, r.title, r.page, r.section, r.text, float(r.score)) for r in rows]

    rows = db.execute(
        select(DocumentChunk, KnowledgeSource.title)
        .join(KnowledgeSource, KnowledgeSource.id == DocumentChunk.source_id)
        .where(DocumentChunk.tenant_id == tenant_id, KnowledgeSource.tenant_id == tenant_id,
               KnowledgeSource.status == "ready")
    ).all()
    scored = sorted(
        (ContextPassage(c.source_id, title, c.page, c.section, c.text, _cosine(qvec, c.embedding))
         for c, title in rows),
        key=lambda p: p.score, reverse=True,
    )
    return scored[:top_k]
