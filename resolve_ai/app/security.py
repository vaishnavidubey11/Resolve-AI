"""Password hashing (scrypt) and signed JWT tokens."""
import base64
import hashlib
import hmac
import os
import re
from datetime import datetime, timedelta, timezone

import jwt

from app.config import Settings

_N, _R, _P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, dk_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(dk_b64)
        dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# Used to keep login timing similar for unknown users.
DUMMY_HASH = hash_password("not-a-real-password")


def create_token(settings: Settings, claims: dict, minutes: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {**claims, "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(settings: Settings, token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        return None


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "tenant"
