from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.deps import Principal, get_db, get_principal, require_staff
from app.errors import not_found
from app.models import Ticket
from app.schemas import Page, TicketOut, TicketPatch

router = APIRouter(prefix="/tickets", tags=["tickets"])


def _load(db: Session, p: Principal, ticket_id: str) -> Ticket:
    t = db.get(Ticket, ticket_id)
    if t is None or t.tenant_id != p.tenant_id:
        raise not_found("Ticket")
    if p.kind == "customer" and t.conversation_id != p.conversation_id:
        raise not_found("Ticket")
    return t


@router.get("", response_model=Page[TicketOut])
def list_tickets(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                 p: Principal = Depends(require_staff), db: Session = Depends(get_db)):
    q = select(Ticket).where(Ticket.tenant_id == p.tenant_id)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    items = db.scalars(q.order_by(Ticket.created_at.desc()).limit(limit).offset(offset)).all()
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get("/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: str, p: Principal = Depends(get_principal), db: Session = Depends(get_db)):
    return _load(db, p, ticket_id)


@router.patch("/{ticket_id}", response_model=TicketOut)
def update_ticket(ticket_id: str, body: TicketPatch, p: Principal = Depends(require_staff),
                  db: Session = Depends(get_db)):
    t = _load(db, p, ticket_id)
    t.status = body.status
    db.commit()
    return t
