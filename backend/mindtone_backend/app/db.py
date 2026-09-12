"""Database layer. Same schema as the Streamlit version (users, checkins)
minus the removed Assessment table. Uses FastAPI's dependency-injection
pattern for sessions instead of manual open/close."""
import datetime

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from . import settings

connect_args = {'check_same_thread': False} if settings.DATABASE_URL.startswith('sqlite') else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
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

    checkins = relationship('CheckIn', back_populates='user', cascade='all, delete-orphan')


class CheckIn(Base):
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


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    """FastAPI dependency: yields a session, always closes it after the
    request, even if the handler raises."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_by_username(db, username: str):
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db, email: str):
    return db.query(User).filter(User.email == email).first()


def create_user(db, username: str, email: str, password_hash: str) -> User:
    user = User(username=username, email=email, password_hash=password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def record_failed_login(db, user: User):
    user.failed_attempts = (user.failed_attempts or 0) + 1
    if user.failed_attempts >= MAX_FAILED_LOGIN_ATTEMPTS:
        user.locked_until = datetime.datetime.utcnow() + datetime.timedelta(minutes=LOCKOUT_MINUTES)
    db.commit()


def reset_failed_logins(db, user: User):
    user.failed_attempts = 0
    user.locked_until = None
    db.commit()


def is_locked_out(user: User) -> bool:
    return user.locked_until is not None and user.locked_until > datetime.datetime.utcnow()


def update_password(db, user: User, new_password_hash: str):
    user.password_hash = new_password_hash
    db.commit()


def insert_checkin(db, user_id: int, date: datetime.date, prompt_sentence: str,
                    emotion_result: dict) -> CheckIn:
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
    db.add(checkin)
    db.commit()
    db.refresh(checkin)
    return checkin


def get_checkins_for_date(db, user_id: int, date: datetime.date):
    return (db.query(CheckIn)
            .filter(CheckIn.user_id == user_id, CheckIn.date == date)
            .order_by(CheckIn.created_at.asc())
            .all())


def get_user_checkins(db, user_id: int, since: datetime.date = None):
    q = db.query(CheckIn).filter(CheckIn.user_id == user_id)
    if since is not None:
        q = q.filter(CheckIn.date >= since)
    return q.order_by(CheckIn.date.asc(), CheckIn.created_at.asc()).all()
