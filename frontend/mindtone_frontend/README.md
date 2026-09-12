# MindTone Frontend (React + Vite + Tailwind)

## Local setup

```bash
npm install
cp .env.example .env
# edit .env: VITE_API_BASE_URL should point at your running backend
npm run dev
```

Open http://localhost:5173. Make sure the backend is running first
(see `mindtone_backend/README.md`) and its `CORS_ORIGINS` includes
`http://localhost:5173`.

## Deploying to Vercel

1. Push this folder to a GitHub repo.
2. Go to vercel.com → New Project → import the repo.
3. Framework preset: Vite (should auto-detect).
4. Add environment variable: `VITE_API_BASE_URL` = your deployed backend URL
   (e.g. `https://mindtone-backend.onrender.com`)
5. Deploy.

After deploying, update your **backend's** `CORS_ORIGINS` environment
variable to include your new Vercel URL (e.g.
`https://mindtone.vercel.app`), or the frontend will get CORS errors on
every request.

## Pages

- `/` — landing page (login/signup) if logged out, dashboard if logged in
- `/checkin` — daily mood check-in (mic recording or file upload)
- `/history` — long-term pattern detection + emotion trend charts
- `/resources` — support resources
- `/account` — profile + change password

## Notes

- Microphone recording uses the browser's MediaRecorder API — requires
  HTTPS in production (Vercel provides this automatically) or
  `localhost` for local dev; it won't work over plain HTTP on a remote host.
- Auth token is stored in `localStorage` and attached to every API
  request automatically (see `src/api/client.js`).
