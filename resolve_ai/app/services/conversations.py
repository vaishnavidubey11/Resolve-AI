from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Conversation, Message


def add_message(db: Session, conv: Conversation, role: str, content: str, status: str | None = None,
                answer_type: str | None = None, sources: list | None = None) -> Message:
    seq = (db.scalar(select(func.max(Message.seq)).where(Message.conversation_id == conv.id)) or 0) + 1
    msg = Message(conversation_id=conv.id, seq=seq, role=role, content=content, status=status,
                  answer_type=answer_type, sources=sources or [])
    db.add(msg)
    db.flush()
    return msg


def recent_messages(db: Session, conv: Conversation, limit: int) -> list[Message]:
    rows = db.scalars(
        select(Message).where(Message.conversation_id == conv.id).order_by(Message.seq.desc()).limit(limit)
    ).all()
    return list(reversed(rows))
