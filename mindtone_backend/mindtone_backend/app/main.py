"""MindTone API. Run locally with:
    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db, model_loader, settings
from .routers import account_router, auth_router, checkin_router, patterns_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    model_loader.load_model()  # loads once at startup, not on the first request
    yield


app = FastAPI(title='MindTone API', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(auth_router.router)
app.include_router(checkin_router.router)
app.include_router(patterns_router.router)
app.include_router(account_router.router)


@app.get('/api/health')
def health():
    return {'status': 'ok'}
