# Render (and most platforms) set $PORT dynamically and expect the
# container to listen on it. Defaults to 8080 for local/self-hosted use
# where $PORT isn't set.
FROM python:3.11-slim

# libsndfile1 is required by the `soundfile` package (used for audio
# decoding in ser_pipeline/data/audio.py) — without it, soundfile fails
# to import at all, not just at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# CPU-only torch/torchaudio — the default PyPI wheels bundle CUDA
# runtime libraries (several hundred MB, some of which get mapped into
# memory even when unused), which matters a lot on a 512MB free-tier host.
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No persistent disk on most free-tier hosts — /code/app/model_data is
# recreated fresh on every restart, so the model checkpoint and backbone
# weights re-download each time the container restarts. That's expected
# and unavoidable on most free tiers; it just means the first request
# after a cold start is slow.
RUN mkdir -p /code/app/model_data

EXPOSE 8080

# Shell form (not exec array form) so $PORT actually gets substituted.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
