from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import auth, db, schemas

router = APIRouter(prefix='/api/account', tags=['account'])


@router.get('/me', response_model=schemas.UserResponse)
def get_me(current_user: db.User = Depends(auth.get_current_user)):
    return current_user


@router.post('/change-password')
def change_password(
    payload: schemas.ChangePasswordRequest,
    current_user: db.User = Depends(auth.get_current_user),
    session: Session = Depends(db.get_db),
):
    pw_error = auth.validate_password_strength(payload.new_password)
    if pw_error:
        raise HTTPException(status_code=400, detail=pw_error)
    if not auth.verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail='Current password is incorrect.')
    db.update_password(session, current_user, auth.hash_password(payload.new_password))
    return {'detail': 'Password updated.'}
