"""SQLAlchemy models (SRS section 6.1).

The ``EmbeddingType`` uses pgvector on PostgreSQL and falls back to a JSON array on
other dialects (SQLite is used for fast, dependency-free tests).
"""
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator, UserDefinedType

from app.config import get_settings
from app.db import Base


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _PgVector(UserDefinedType):
    cache_ok = True

    def __init__(self, dim: int):
        self.dim = dim

    def get_col_spec(self, **kw):
        return f"vector({self.dim})"


class EmbeddingType(TypeDecorator):
    impl = Text
    cache_ok = True

    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim or get_settings().embedding_dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PgVector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return "[" + ",".join(f"{float(x):.7f}" for x in value) + "]"
        return json.dumps(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
        text = str(value).strip()
        if dialect.name == "postgresql":  # pgvector text form: "[0.1,0.2,...]"
            return [float(x) for x in text.strip("[]").split(",") if x]
        return json.loads(text)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ten"))
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)


class User(Base, TimestampMixin):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("usr"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    email: Mapped[str] = mapped_column(String(320))
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(20), default="admin")  # admin | operator


class KnowledgeSource(Base, TimestampMixin):
    __tablename__ = "knowledge_sources"
    __table_args__ = (UniqueConstraint("tenant_id", "checksum", name="uq_source_tenant_checksum"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("src"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    filename: Mapped[str] = mapped_column(String(300))
    source_type: Mapped[str] = mapped_column(String(20))  # pdf | text
    status: Mapped[str] = mapped_column(String(20), default="processing")  # processing|ready|failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    checksum: Mapped[str] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="source", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (Index("ix_chunks_tenant_source", "tenant_id", "source_id"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("chk"))
    source_id: Mapped[str] = mapped_column(ForeignKey("knowledge_sources.id", ondelete="CASCADE"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(300), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingType())
    source: Mapped[KnowledgeSource] = relationship(back_populates="chunks")


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("conv"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    customer_ref: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="open")  # open|escalated|closed
    context: Mapped[dict] = mapped_column(JSON, default=dict)  # pending tool-flow state
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True,
        order_by="Message.seq",
    )


class Message(Base, TimestampMixin):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("conversation_id", "seq", name="uq_message_seq"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("msg"))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)  # strict chronological ordering
    role: Mapped[str] = mapped_column(String(20))  # customer | assistant
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    answer_type: Mapped[str | None] = mapped_column(String(20), nullable=True)  # document|order_api|system
    sources: Mapped[list] = mapped_column(JSON, default=list)
    feedback: Mapped[str | None] = mapped_column(String(20), nullable=True)  # helpful | not_helpful
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Ticket(Base, TimestampMixin):
    __tablename__ = "tickets"
    __table_args__ = (UniqueConstraint("conversation_id", name="uq_ticket_conversation"),)  # idempotency
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("tkt"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    summary: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(40), default="general")
    priority: Mapped[str] = mapped_column(String(20), default="normal")
    status: Mapped[str] = mapped_column(String(20), default="open")  # open|in_progress|resolved


class ToolExecution(Base, TimestampMixin):
    __tablename__ = "tool_executions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("tool"))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    tool_name: Mapped[str] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(20))  # success|denied|not_found|error
    result_reference: Mapped[str | None] = mapped_column(String(100), nullable=True)
