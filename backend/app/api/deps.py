"""
Shared FastAPI dependencies.

`get_current_user` is the replacement for the old
`x_user_id: str = Header(...)` pattern used throughout rfp.py / chat.py.
Instead of trusting a client-supplied header, it verifies a signed JWT
issued at login and returns the user id encoded inside it. A request with
no token, an expired token, or a tampered token is rejected with 401
before any route code runs.
"""

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.security import decode_access_token

_bearer_scheme = HTTPBearer(
    auto_error=True,
    description="Paste the access token returned by /auth/login or /auth/signup.",
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """Returns the authenticated user's id (str) or raises HTTP 401."""
    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id