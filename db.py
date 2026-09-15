"""Database layer. Defaults to a local SQLite file for easy local dev.
Set DATABASE_URL (e.g. a Supabase/Render Postgres URL) for real deployment
— SQLite files on most free hosts are wiped on every redeploy/restart,
which would silently erase weeks of tracked history.

SCHEMA CHANGE from v3: the Assessment table (PHQ-9/GAD-7/WHO-5 forms) has
been removed — replaced with pattern detection computed directly from
accumulated CheckIn history (see patterns.py). If you have an existing
v3 database, drop all tables before redeploying (e.g. `DROP TABLE
checkins; DROP TABLE users; DROP TABLE assessments;` in Supabase's SQL
editor) so they get recreated with the current schema — this will lose
existing data.
"""
import datetime
import os

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./mh_tracker.db')

connect_args = {'check_same_thread': False} if DATABASE_URL.startswith('sqlite') else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
# expire_on_commit=False: without this, SQLAlchemy expires every loaded
# attribute on an object after any commit — fine if you always re-read
# attributes before the session closes, but several call sites here
# return an ORM object AFTER closing their session (e.g. login() returns
# the User to app.py, which reads user.id afterward). Without this flag,
# any commit in between (e.g. resetting a login-lockout counter) leaves
# the object detached-and-expired, and that later attribute read raises
# DetachedInstanceError. This was an actual production bug — see the
# regression test in test_db_auth.py if present.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False,
                             expire_on_commit=False)
Base = declarative_base()

MAX_FAILED_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 10


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    failed_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)

    checkins = relationship('CheckIn', back_populates='user',
                             cascade='all, delete-orphan')


class CheckIn(Base):
    """A daily voice mood check-in — emotion detection ONLY. Long-term
    pattern analysis (see patterns.py) is computed on top of a history of
    these, not derived per-entry."""
    __tablename__ = 'checkins'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    prompt_sentence = Column(String(256))

    predicted_emotion = Column(String(16))
    confidence = Column(Float)
    uncertainty = Column(Float)
    prob_angry = Column(Float)
    prob_disgust = Column(Float)
    prob_fear = Column(Float)
    prob_happy = Column(Float)
    prob_neutral = Column(Float)
    prob_sad = Column(Float)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship('User', back_populates='checkins')

    # No unique constraint on (user_id, date) — multiple check-ins/day allowed.


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()


def get_user_by_username(session, username: str):
    return session.query(User).filter(User.username == username).first()


def get_user_by_email(session, email: str):
    return session.query(User).filter(User.email == email).first()


def create_user(session, username: str, email: str, password_hash: str) -> User:
    user = User(username=username, email=email, password_hash=password_hash)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def record_failed_login(session, user: User):
    user.failed_attempts = (user.failed_attempts or 0) + 1
    if user.failed_attempts >= MAX_FAILED_LOGIN_ATTEMPTS:
        user.locked_until = datetime.datetime.utcnow() + datetime.timedelta(minutes=LOCKOUT_MINUTES)
    session.commit()


def reset_failed_logins(session, user: User):
    user.failed_attempts = 0
    user.locked_until = None
    session.commit()


def is_locked_out(user: User) -> bool:
    return user.locked_until is not None and user.locked_until > datetime.datetime.utcnow()


def update_password(session, user: User, new_password_hash: str):
    user.password_hash = new_password_hash
    session.commit()


def insert_checkin(session, user_id: int, date: datetime.date, prompt_sentence: str,
                    emotion_result: dict) -> CheckIn:
    """Always inserts a new row — multiple check-ins on the same day are
    all kept, not overwritten. Stores emotion only, no derived MH score."""
    probs = emotion_result['class_probabilities']
    checkin = CheckIn(
        user_id=user_id, date=date, prompt_sentence=prompt_sentence,
        predicted_emotion=emotion_result['predicted_emotion'],
        confidence=emotion_result['confidence'],
        uncertainty=emotion_result['uncertainty'],
        prob_angry=probs.get('Angry', 0.0),
        prob_disgust=probs.get('Disgust', 0.0),
        prob_fear=probs.get('Fear', 0.0),
        prob_happy=probs.get('Happy', 0.0),
        prob_neutral=probs.get('Neutral', 0.0),
        prob_sad=probs.get('Sad', 0.0),
    )
    session.add(checkin)
    session.commit()
    session.refresh(checkin)
    return checkin


def get_checkins_for_date(session, user_id: int, date: datetime.date):
    return (session.query(CheckIn)
            .filter(CheckIn.user_id == user_id, CheckIn.date == date)
            .order_by(CheckIn.created_at.asc())
            .all())


def get_user_checkins(session, user_id: int, since: datetime.date = None):
    q = session.query(CheckIn).filter(CheckIn.user_id == user_id)
    if since is not None:
        q = q.filter(CheckIn.date >= since)
    return q.order_by(CheckIn.date.asc(), CheckIn.created_at.asc()).all()
