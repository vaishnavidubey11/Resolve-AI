from dataclasses import dataclass
from typing import Iterator

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.errors import not_found
from app.models import Conversation, User
from app.security import decode_token

bearer = HTTPBearer(auto_error=False)


def get_db(request: Request) -> Iterator[Session]:
    db = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()


@dataclass
class Principal:
    kind: str  # staff | customer
    tenant_id: str
    user: User | None = None
    conversation_id: str | None = None

    @property
    def role(self) -> str | None:
        return self.user.role if self.user else None


def get_principal(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                  db: Session = Depends(get_db)) -> Principal:
    unauthorized = HTTPException(401, "Authentication required.", headers={"WWW-Authenticate": "Bearer"})
    if creds is None:
        raise unauthorized
    claims = decode_token(request.app.state.settings, creds.credentials)
    if not claims:
        raise unauthorized
    if claims.get("typ") == "staff":
        user = db.get(User, claims.get("sub"))
        # tenant comes from the database record, never from client-supplied data
        if user is None:
            raise unauthorized
        return Principal("staff", user.tenant_id, user=user)
    if claims.get("typ") == "customer" and claims.get("cid") and claims.get("tid"):
        return Principal("customer", claims["tid"], conversation_id=claims["cid"])
    raise unauthorized


def require_staff(p: Principal = Depends(get_principal)) -> Principal:
    if p.kind != "staff":
        raise HTTPException(403, "You do not have permission to perform this action.")
    return p


def require_admin(p: Principal = Depends(require_staff)) -> Principal:
    if p.role != "admin":
        raise HTTPException(403, "Administrator permission is required.")
    return p


def load_conversation(db: Session, p: Principal, conversation_id: str) -> Conversation:
    """Tenant- and participant-scoped lookup. Other tenants' ids are indistinguishable from missing ones."""
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.tenant_id != p.tenant_id:
        raise not_found("Conversation")
    if p.kind == "customer" and p.conversation_id != conv.id:
        raise not_found("Conversation")
    return conv


def rate_limit(name: str, setting_attr: str):
    def dep(request: Request) -> None:
        s = request.app.state.settings
        if not s.rate_limit_enabled:
            return
        from app.middleware import client_key
        request.app.state.limiter.check(f"{name}:{client_key(request)}", getattr(s, setting_attr))
    return dep
