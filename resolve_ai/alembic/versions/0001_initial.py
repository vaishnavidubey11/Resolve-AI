"""Initial schema: tenants, users, knowledge sources, chunks (pgvector), conversations, messages, tickets, tool executions.

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

from app.models import EmbeddingType

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

ts = lambda: sa.Column("created_at", sa.DateTime(timezone=True), nullable=False)  # noqa: E731


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table("tenants",
        sa.Column("id", sa.String(40), primary_key=True), sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(80), nullable=False), ts())
    op.create_index("ix_tenants_slug", "tenants", ["slug"], unique=True)

    op.create_table("users",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("tenant_id", sa.String(40), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(320), nullable=False), sa.Column("password_hash", sa.String(300), nullable=False),
        sa.Column("role", sa.String(20), nullable=False), ts(), sa.UniqueConstraint("email", name="uq_users_email"))
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])

    op.create_table("knowledge_sources",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("tenant_id", sa.String(40), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(300), nullable=False), sa.Column("filename", sa.String(300), nullable=False),
        sa.Column("source_type", sa.String(20), nullable=False), sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error", sa.Text), sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default="0"), ts(),
        sa.UniqueConstraint("tenant_id", "checksum", name="uq_source_tenant_checksum"))
    op.create_index("ix_knowledge_sources_tenant_id", "knowledge_sources", ["tenant_id"])

    op.create_table("document_chunks",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("source_id", sa.String(40), sa.ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", sa.String(40), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False), sa.Column("text", sa.Text, nullable=False),
        sa.Column("page", sa.Integer), sa.Column("section", sa.String(300)),
        sa.Column("embedding", EmbeddingType(), nullable=False))
    op.create_index("ix_chunks_tenant_source", "document_chunks", ["tenant_id", "source_id"])
    # NOTE: for large corpora add an ANN index, e.g.
    #   CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);   (needs dim <= 2000)

    op.create_table("conversations",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("tenant_id", sa.String(40), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("customer_ref", sa.String(200)), sa.Column("status", sa.String(20), nullable=False),
        sa.Column("context", sa.JSON, nullable=False), ts(),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_conversations_tenant_id", "conversations", ["tenant_id"])

    op.create_table("messages",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("conversation_id", sa.String(40), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer, nullable=False), sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False), sa.Column("status", sa.String(30)),
        sa.Column("answer_type", sa.String(20)), sa.Column("sources", sa.JSON, nullable=False),
        sa.Column("feedback", sa.String(20)), ts(), sa.UniqueConstraint("conversation_id", "seq", name="uq_message_seq"))
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])

    op.create_table("tickets",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("tenant_id", sa.String(40), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("conversation_id", sa.String(40), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("summary", sa.Text, nullable=False), sa.Column("category", sa.String(40), nullable=False),
        sa.Column("priority", sa.String(20), nullable=False), sa.Column("status", sa.String(20), nullable=False), ts(),
        sa.UniqueConstraint("conversation_id", name="uq_ticket_conversation"))
    op.create_index("ix_tickets_tenant_id", "tickets", ["tenant_id"])

    op.create_table("tool_executions",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("conversation_id", sa.String(40), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tool_name", sa.String(60), nullable=False), sa.Column("status", sa.String(20), nullable=False),
        sa.Column("result_reference", sa.String(100)), ts())
    op.create_index("ix_tool_executions_conversation_id", "tool_executions", ["conversation_id"])


def downgrade() -> None:
    for t in ("tool_executions", "tickets", "messages", "conversations", "document_chunks",
              "knowledge_sources", "users", "tenants"):
        op.drop_table(t)
