# MindTone Backend (FastAPI)

REST API for the MindTone app: JWT auth, daily voice check-ins (runs the
SER model), long-term mood pattern detection.

## Local setup

```bash
pip install -r requirements.txt
```

Place your trained `ser_model.pth` at `app/model_data/ser_model.pth`, or
set `MODEL_URL` to a direct-download link (e.g. Hugging Face Hub) to
have it auto-download on startup.

```bash
cp .env.example .env
# edit .env — for local dev you can leave DATABASE_URL and MODEL_URL blank
uvicorn app.main:app --reload
```

API docs (interactive): http://localhost:8000/docs

## Environment variables

- `DATABASE_URL` — Postgres connection string for production (defaults
  to local SQLite, which most free hosts wipe on redeploy — don't use
  SQLite in production)
- `JWT_SECRET_KEY` — a real random secret in production (the default is
  insecure and only for local dev)
- `CORS_ORIGINS` — comma-separated list of allowed frontend origins,
  e.g. `https://mindtone.vercel.app,http://localhost:5173`
- `MODEL_URL` — direct-download URL for `ser_model.pth`

## Deploying to Render

1. Push this folder to a GitHub repo.
2. Render → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables: `DATABASE_URL`, `JWT_SECRET_KEY`, `CORS_ORIGINS`, `MODEL_URL`
6. Deploy. First request will download the pretrained backbone from
   Hugging Face plus your checkpoint — expect the first check-in after a
   deploy to be slow.

Get a free Postgres `DATABASE_URL` from [Supabase](https://supabase.com)
— same as before, don't skip this for production.

## API endpoints

- `POST /api/auth/signup` / `POST /api/auth/login` — returns a JWT
- `GET /api/checkin/prompt?exclude=...` — a random prompt sentence
- `POST /api/checkin?prompt_sentence=...` (multipart form, `audio` file) — analyze + save
- `GET /api/checkin/today` / `GET /api/checkin/history?days=90`
- `GET /api/patterns?days=30` — long-term mood pattern summary
- `GET /api/account/me` / `POST /api/account/change-password`

All endpoints except signup/login/prompt require `Authorization: Bearer <token>`.
