import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import auth, db, schemas

router = APIRouter(prefix='/api/auth', tags=['auth'])


@router.post('/signup', response_model=schemas.TokenResponse)
def signup(payload: schemas.SignupRequest, session: Session = Depends(db.get_db)):
    pw_error = auth.validate_password_strength(payload.password)
    if pw_error:
        raise HTTPException(status_code=400, detail=pw_error)

    if db.get_user_by_username(session, payload.username) is not None:
        raise HTTPException(status_code=400, detail='That username is already taken.')
    if db.get_user_by_email(session, payload.email) is not None:
        raise HTTPException(status_code=400, detail='An account with that email already exists.')

    user = db.create_user(session, payload.username, payload.email,
                            auth.hash_password(payload.password))
    token = auth.create_access_token(user.id, user.username)
    return schemas.TokenResponse(access_token=token, username=user.username)


@router.post('/login', response_model=schemas.TokenResponse)
def login(payload: schemas.LoginRequest, session: Session = Depends(db.get_db)):
    user = db.get_user_by_username(session, payload.username.strip())
    if user is None:
        raise HTTPException(status_code=401, detail='Incorrect username or password.')

    if db.is_locked_out(user):
        minutes_left = max(1, int((user.locked_until - datetime.datetime.utcnow())
                                   .total_seconds() // 60) + 1)
        raise HTTPException(status_code=429,
                             detail=f'Too many failed attempts. Try again in about {minutes_left} minute(s).')

    if not auth.verify_password(payload.password, user.password_hash):
        db.record_failed_login(session, user)
        raise HTTPException(status_code=401, detail='Incorrect username or password.')

    db.reset_failed_logins(session, user)
    token = auth.create_access_token(user.id, user.username)
    return schemas.TokenResponse(access_token=token, username=user.username)
