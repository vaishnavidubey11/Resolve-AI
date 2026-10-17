"""Ticket creation / human escalation (FR-018, FR-019). Idempotent per conversation."""
import re

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Conversation, Message, Ticket

URGENT_RE = re.compile(r"\b(urgent|asap|immediately|emergency)\b", re.I)
CATEGORY_RULES = [
    ("order", re.compile(r"\b(order|package|parcel|delivery|shipping|tracking)\b", re.I)),
    ("returns", re.compile(r"\b(return|refund|exchange|replace)\w*\b", re.I)),
    ("product", re.compile(r"\b(product|size|warranty|broken|defect\w*)\b", re.I)),
]


def _summarize(db: Session, conv: Conversation, reason: str | None) -> tuple[str, str, str]:
    customer_msgs = [m.content for m in conv.messages if m.role == "customer"]
    last = customer_msgs[-1] if customer_msgs else ""
    blob = " ".join(customer_msgs[-3:])
    category = next((name for name, rx in CATEGORY_RULES if rx.search(blob)), "general")
    priority = "high" if URGENT_RE.search(blob) else "normal"
    summary = (reason or "Customer needs human assistance.") + f" Latest customer message: {last[:300]}"
    return summary, category, priority


def escalate_conversation(db: Session, conv: Conversation, reason: str | None = None) -> tuple[Ticket, bool]:
    """Create (or return the existing) ticket for a conversation and mark it escalated."""
    existing = db.scalar(select(Ticket).where(Ticket.conversation_id == conv.id))
    if existing:
        conv.status = "escalated" if conv.status != "closed" else conv.status
        return existing, False
    summary, category, priority = _summarize(db, conv, reason)
    ticket = Ticket(tenant_id=conv.tenant_id, conversation_id=conv.id, summary=summary,
                    category=category, priority=priority)
    try:
        with db.begin_nested():
            db.add(ticket)
            db.flush()
    except IntegrityError:  # concurrent escalation: unique(conversation_id) protects against duplicates
        return db.scalar(select(Ticket).where(Ticket.conversation_id == conv.id)), False
    conv.status = "escalated"
    return ticket, True
