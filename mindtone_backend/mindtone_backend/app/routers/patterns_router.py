import datetime

import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .. import auth, db, patterns, schemas

router = APIRouter(prefix='/api/patterns', tags=['patterns'])


@router.get('', response_model=schemas.PatternSummary)
def get_patterns(
    days: int | None = Query(default=30),
    current_user: db.User = Depends(auth.get_current_user),
    session: Session = Depends(db.get_db),
):
    since = (datetime.date.today() - datetime.timedelta(days=days)) if days else None
    checkins = db.get_user_checkins(session, current_user.id, since=since)

    if not checkins:
        return schemas.PatternSummary(insufficient_data=True, n_checkins=0,
                                        min_required=patterns.MIN_CHECKINS_FOR_PATTERN)

    df = pd.DataFrame([{'date': c.date, 'emotion': c.predicted_emotion} for c in checkins])
    result = patterns.summarize_mood_patterns(df, window=min(len(df), days or len(df)))

    if result['insufficient_data']:
        return schemas.PatternSummary(insufficient_data=True, n_checkins=result['n_checkins'],
                                        min_required=result['min_required'])

    return schemas.PatternSummary(
        insufficient_data=False,
        n_checkins=result['n_checkins'],
        emotion_pct=result['emotion_pct'],
        low_mood_pct=result['low_mood_pct'],
        anxious_pct=result['anxious_pct'],
        tense_pct=result['tense_pct'],
        positive_pct=result['positive_pct'],
        flags=[schemas.PatternFlag(**f) for f in result['flags']],
        trend=result['trend'],
        trend_delta=result['trend_delta'],
    )
