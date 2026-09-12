import datetime

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from .. import auth, db, model_loader, prompts, schemas

router = APIRouter(prefix='/api/checkin', tags=['checkin'])

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'}
MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20MB — generous for a few seconds of speech


@router.get('/prompt', response_model=schemas.PromptResponse)
def get_prompt(exclude: str = Query(default=None)):
    return schemas.PromptResponse(sentence=prompts.get_random_prompt(exclude=exclude))


@router.post('', response_model=schemas.CheckInResponse)
async def submit_checkin(
    prompt_sentence: str,
    audio: UploadFile = File(...),
    current_user: db.User = Depends(auth.get_current_user),
    session: Session = Depends(db.get_db),
):
    ext = '.' + audio.filename.rsplit('.', 1)[-1].lower() if '.' in audio.filename else '.wav'
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f'Unsupported audio format: {ext}')

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail='Audio file too large.')
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail='Empty audio file.')

    try:
        emotion_result = model_loader.analyze_audio_bytes(audio_bytes, suffix=ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Audio analysis failed: {e}')

    checkin = db.insert_checkin(session, current_user.id, datetime.date.today(),
                                  prompt_sentence, emotion_result)
    return checkin


@router.get('/today', response_model=list[schemas.CheckInResponse])
def get_today_checkins(
    current_user: db.User = Depends(auth.get_current_user),
    session: Session = Depends(db.get_db),
):
    return db.get_checkins_for_date(session, current_user.id, datetime.date.today())


@router.get('/history', response_model=list[schemas.CheckInResponse])
def get_history(
    days: int | None = Query(default=90, description='Look back this many days; omit for all-time'),
    current_user: db.User = Depends(auth.get_current_user),
    session: Session = Depends(db.get_db),
):
    since = (datetime.date.today() - datetime.timedelta(days=days)) if days else None
    return db.get_user_checkins(session, current_user.id, since=since)
