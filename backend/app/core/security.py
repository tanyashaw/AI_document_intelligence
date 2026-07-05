"""
Password hashing and JWT helpers.

This replaces the old "trust whatever X-User-Id header the client sends"
scheme with real authentication: a user proves who they are once with a
password, and every subsequent request proves it again with a signed,
expiring token that only this server could have issued.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
)


# ── Passwords ─────────────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), password_hash.encode("utf-8")
        )
    except ValueError:
        # Malformed hash — never crash auth on this, just fail closed.
        return False


# ── JWT access tokens ─────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises jwt.PyJWTError (ExpiredSignatureError, InvalidTokenError, ...)
    on any invalid/expired/tampered token. Callers turn that into a 401."""
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])