# Mental Health Voice Tracker

A Streamlit web app: users sign up, log in, and do a daily voice
check-in (read a fixed sentence aloud). Speech emotion recognition runs
on the recording, converts it into heuristic mental-health indicators
(Depression / Anxiety / Stress / Emotional Wellbeing), and stores the
result so trends can be tracked over weeks.

Built on top of the [SER pipeline](./ser_pipeline) (locked-in result:
test WA=71.46%/UA=70.63%) — see `ser_pipeline/README.md` for the model
itself and its training.

## How it works

- **Login/signup** (`app.py`) — bcrypt-hashed passwords, SQLAlchemy-backed.
- **Daily Check-in** (`pages/1_Daily_Checkin.py`) — shows the same fixed
  sentence to everyone on a given calendar day (consistent content makes
  emotion trends comparable day to day), records or accepts an uploaded
  audio file, runs it through the model, saves the result.
- **My History** (`pages/2_My_History.py`) — trend charts, a sustained-
  pattern check over the last 7 check-ins, and a full history table,
  filterable by time range (7/30/90 days or all time).

Audio itself is **not stored** — only the derived scores. If you want to
keep clips too, that's a deliberate change to make (see "Extending" below).

## Local setup

```bash
pip install -r requirements.txt
```

Place your trained `ser_model.pth` at `model_data/ser_model.pth`
(relative to this folder), or set `MODEL_URL` in `.env` to have it
auto-download on first run (see below).

```bash
cp .env.example .env
# edit .env if needed — both fields can stay blank for local testing
# (uses local SQLite + a manually-placed model_data/ser_model.pth)

streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Before deploying: get the model checkpoint somewhere downloadable

`ser_model.pth` is ~385MB — don't commit it to git. Recommended: upload
it to a free Hugging Face Hub model repo.

```bash
pip install huggingface_hub
huggingface-cli login          # paste a token from huggingface.co/settings/tokens
huggingface-cli repo create your-username/ser-model-checkpoint --type model
huggingface-cli upload your-username/ser-model-checkpoint ser_model.pth
```

Your `MODEL_URL` is then:
```
https://huggingface.co/your-username/ser-model-checkpoint/resolve/main/ser_model.pth
```

## Before deploying: get a persistent database

**Don't rely on SQLite in production.** Most free hosts (Render,
Streamlit Community Cloud, Railway) wipe the local filesystem on every
redeploy or restart — that would silently erase weeks of check-in
history. Use a real hosted Postgres instead:

- **[Supabase](https://supabase.com)** (recommended) — free tier
  includes a persistent Postgres database. Create a project, copy the
  connection string from Settings -> Database, set it as `DATABASE_URL`.
- Render also offers a free Postgres tier (time-limited) if you'd
  rather keep everything on one platform.

`DATABASE_URL` format: `postgresql://user:password@host:5432/dbname`

## Deploying

### Option A — Streamlit Community Cloud (simplest, purpose-built for this)

1. Push this folder to a GitHub repo (excluding `model_data/` and `.env`
   — add both to `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the
   repo, set the main file to `app.py`.
3. In the app's Settings -> Secrets, add:
   ```toml
   DATABASE_URL = "postgresql://..."
   MODEL_URL = "https://huggingface.co/..."
   ```
4. Deploy. First request will download the model checkpoint (~385MB)
   and the pretrained backbone from Hugging Face — expect the first
   check-in after a deploy to be slow.

### Option B — Render (as a single web service)

1. Push to GitHub as above.
2. New -> Web Service -> connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Add `DATABASE_URL` and `MODEL_URL` as environment variables in the
   Render dashboard.
6. **Note:** Render's free tier has limited RAM (check current limits on
   their pricing page) — the model (~94M params, WavLM-base) fits, but
   headroom is tight alongside Streamlit + torch + transformers.
   If it OOMs on the free tier, the paid Starter tier resolves it.

Vercel/Netlify **won't work** for this — they only serve static sites
and serverless functions, not a persistent Python/Streamlit process.

## Privacy notes

- Passwords are bcrypt-hashed, never stored in plain text.
- Raw audio is discarded immediately after analysis by default — only
  derived scores are persisted.
- The mental-health scores are **heuristic**, hand-set weighted
  combinations of emotion probabilities — not learned from labeled
  mental-health data, and not a diagnosis. The app's own copy reflects
  this; keep it that way if you extend the UI.
- If you deploy this for real users (not just a demo/project), think
  through consent, data retention, and what happens if the sustained-
  pattern flags trigger for someone who isn't just testing the app —
  at minimum, the app should point them to real support resources, not
  just display a score.

## Extending

- **Keep audio clips**: store the uploaded bytes (e.g. to S3/Supabase
  Storage) alongside the `CheckIn` row instead of discarding them in
  `model_loader.analyze_audio_bytes`.
- **Email reminders**: a daily cron (e.g. a scheduled Render job) that
  queries for users with no check-in today and emails them.
- **Multiple prompts per day** / **free-speech mode**: `prompts.py` and
  the check-in page currently assume one fixed sentence — extending to
  free speech just means skipping the prompt display; the model doesn't
  care what's said.
- **Export**: add a "Download my data" button on the History page
  (`df.to_csv()` from the existing DataFrame).
