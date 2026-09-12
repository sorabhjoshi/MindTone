"""Backend configuration. All secrets come from environment variables —
never hardcode a real SECRET_KEY or DATABASE_URL."""
import os

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./mindtone.db')

# JWT signing key. MUST be set to a real random secret in production —
# the fallback here is only for local dev and is NOT safe to deploy with.
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'dev-only-insecure-secret-change-me')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Comma-separated list of allowed frontend origins for CORS, e.g.
# "https://mindtone.vercel.app,http://localhost:5173"
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5173').split(',')

MODEL_URL = os.environ.get('MODEL_URL', '')  # direct-download URL for ser_model.pth
WORK_DIR = os.environ.get('SER_WORK_DIR', os.path.join(os.path.dirname(__file__), 'model_data'))
os.makedirs(WORK_DIR, exist_ok=True)
os.environ.setdefault('SER_WORK_DIR', WORK_DIR)  # so ser_pipeline/config.py picks it up
