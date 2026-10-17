from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.deps import get_db, rate_limit
from app.models import User
from app.schemas import LoginRequest, TokenResponse
from app.security import DUMMY_HASH, create_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse,
             dependencies=[Depends(rate_limit("login", "login_rate_limit_per_min"))])
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    s = request.app.state.settings
    user = db.scalar(select(User).where(func.lower(User.email) == body.email.lower()))
    ok = verify_password(body.password, user.password_hash if user else DUMMY_HASH)
    if not user or not ok:
        raise HTTPException(401, "Incorrect email or password.")
    token = create_token(s, {"typ": "staff", "sub": user.id}, s.jwt_expire_minutes)
    return TokenResponse(access_token=token, expires_in=s.jwt_expire_minutes * 60,
                         role=user.role, tenant_id=user.tenant_id)
