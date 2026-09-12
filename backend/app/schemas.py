"""Request/response schemas — FastAPI uses these for automatic
validation, so a malformed request (e.g. missing field, wrong type) is
rejected before it ever reaches handler code."""
import datetime

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    username: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class EmotionResult(BaseModel):
    predicted_emotion: str
    confidence: float
    uncertainty: float
    class_probabilities: dict[str, float]


class CheckInResponse(BaseModel):
    id: int
    date: datetime.date
    prompt_sentence: str
    predicted_emotion: str
    confidence: float
    uncertainty: float
    prob_angry: float
    prob_disgust: float
    prob_fear: float
    prob_happy: float
    prob_neutral: float
    prob_sad: float
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class PromptResponse(BaseModel):
    sentence: str


class PatternFlag(BaseModel):
    label: str
    detail: str


class PatternSummary(BaseModel):
    insufficient_data: bool
    n_checkins: int
    min_required: int | None = None
    emotion_pct: dict[str, float] | None = None
    low_mood_pct: float | None = None
    anxious_pct: float | None = None
    tense_pct: float | None = None
    positive_pct: float | None = None
    flags: list[PatternFlag] | None = None
    trend: str | None = None
    trend_delta: float | None = None
