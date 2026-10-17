from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.deps import Principal, get_db, get_principal, load_conversation, rate_limit, require_staff
from app.errors import AppError, not_found
from app.models import Conversation, Message, Tenant, Ticket
from app.schemas import (
    ConversationCreate, ConversationCreated, ConversationOut, ConversationPatch, EscalateRequest,
    FeedbackIn, MessageIn, MessageItem, MessageResponse, Page,
)
from app.security import create_token
from app.services.agent import AgentDeps, handle_message
from app.services.conversations import add_message
from app.services.tickets import escalate_conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationCreated, status_code=201,
             dependencies=[Depends(rate_limit("convcreate", "message_rate_limit_per_min"))])
def create_conversation(body: ConversationCreate, request: Request, db: Session = Depends(get_db)):
    s = request.app.state.settings
    tenant = db.scalar(select(Tenant).where(Tenant.slug == body.tenant))
    if tenant is None:
        raise not_found("Business")
    conv = Conversation(tenant_id=tenant.id, customer_ref=body.customer_ref, context={})
    db.add(conv)
    db.commit()
    token = create_token(s, {"typ": "customer", "cid": conv.id, "tid": tenant.id}, s.customer_token_expire_minutes)
    return ConversationCreated(id=conv.id, status=conv.status, customer_ref=conv.customer_ref,
                               created_at=conv.created_at, updated_at=conv.updated_at, access_token=token)


@router.get("", response_model=Page[ConversationOut])
def list_conversations(status: str | None = Query(None, pattern="^(open|escalated|closed)$"),
                       limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                       p: Principal = Depends(require_staff), db: Session = Depends(get_db)):
    q = select(Conversation).where(Conversation.tenant_id == p.tenant_id)
    if status:
        q = q.where(Conversation.status == status)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    items = db.scalars(q.order_by(Conversation.updated_at.desc()).limit(limit).offset(offset)).all()
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, p: Principal = Depends(get_principal), db: Session = Depends(get_db)):
    return load_conversation(db, p, conversation_id)


@router.patch("/{conversation_id}", response_model=ConversationOut)
def update_status(conversation_id: str, body: ConversationPatch, p: Principal = Depends(require_staff),
                  db: Session = Depends(get_db)):
    conv = load_conversation(db, p, conversation_id)
    conv.status = body.status
    db.commit()
    return conv


@router.get("/{conversation_id}/messages", response_model=Page[MessageItem])
def list_messages(conversation_id: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                  p: Principal = Depends(get_principal), db: Session = Depends(get_db)):
    conv = load_conversation(db, p, conversation_id)
    base = select(Message).where(Message.conversation_id == conv.id)
    total = db.scalar(select(func.count()).select_from(base.subquery()))
    items = db.scalars(base.order_by(Message.seq).limit(limit).offset(offset)).all()
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.post("/{conversation_id}/messages", response_model=MessageResponse,
             dependencies=[Depends(rate_limit("message", "message_rate_limit_per_min"))])
def post_message(conversation_id: str, body: MessageIn, request: Request,
                 p: Principal = Depends(get_principal), db: Session = Depends(get_db)):
    if p.kind != "customer":
        raise AppError(403, "forbidden", "Only the customer session can post customer messages.")
    conv = load_conversation(db, p, conversation_id)
    if conv.status == "closed":
        raise AppError(409, "conversation_closed", "This conversation is closed. Please start a new one.")
    st = request.app.state
    tenant = db.get(Tenant, conv.tenant_id)
    add_message(db, conv, "customer", body.message)
    reply = handle_message(
        AgentDeps(db=db, settings=st.settings, llm=st.llm, embedder=st.embedder, orders=st.order_client, tenant=tenant),
        conv, body.message,
    )
    msg = add_message(db, conv, "assistant", reply.content, reply.status, reply.answer_type, reply.sources)
    ticket = db.scalar(select(Ticket).where(Ticket.conversation_id == conv.id))
    db.commit()
    return MessageResponse(
        conversation_id=conv.id, message_id=msg.id, answer=reply.content, status=reply.status,
        answer_type=reply.answer_type, sources=reply.sources, escalation_available=conv.status != "escalated",
        conversation_status=conv.status, ticket_id=ticket.id if ticket else None,
    )


@router.post("/{conversation_id}/escalate", response_model=dict)
def escalate(conversation_id: str, body: EscalateRequest | None = None, p: Principal = Depends(get_principal),
             db: Session = Depends(get_db)):
    conv = load_conversation(db, p, conversation_id)
    ticket, created = escalate_conversation(db, conv, (body.reason if body else None) or "Escalated by "
                                            + ("customer." if p.kind == "customer" else "support operator."))
    note = (f"I've escalated this to our support team (ticket {ticket.id}). A human agent will follow up."
            if created else f"This conversation is already escalated (ticket {ticket.id}).")
    add_message(db, conv, "assistant", note, "escalated", "system")
    db.commit()
    return {"ticket_id": ticket.id, "created": created, "conversation_status": conv.status, "message": note}


@router.post("/{conversation_id}/messages/{message_id}/feedback", status_code=204)
def feedback(conversation_id: str, message_id: str, body: FeedbackIn, p: Principal = Depends(get_principal),
             db: Session = Depends(get_db)):
    conv = load_conversation(db, p, conversation_id)
    msg = db.get(Message, message_id)
    if msg is None or msg.conversation_id != conv.id or msg.role != "assistant":
        raise not_found("Message")
    msg.feedback = body.value
    db.commit()
