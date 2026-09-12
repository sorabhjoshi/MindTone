"""Auth: bcrypt password hashing (same as the Streamlit version) plus
JWT tokens for stateless API auth — a React SPA can't rely on server-side
session state the way Streamlit did, so the client holds a signed token
and sends it on every request instead."""
import datetime

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from . import db, settings

security = HTTPBearer()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except ValueError:
        return False


def validate_password_strength(password: str):
    import re
    if len(password) < 8:
        return 'Password must be at least 8 characters.'
    if not re.search(r'[A-Za-z]', password):
        return 'Password must include at least one letter.'
    if not re.search(r'\d', password):
        return 'Password must include at least one number.'
    return None


def create_access_token(user_id: int, username: str) -> str:
    payload = {
        'sub': str(user_id),
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token expired')
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(db.get_db),
) -> db.User:
    payload = decode_access_token(credentials.credentials)
    user = session.get(db.User, int(payload['sub']))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User not found')
    return user
