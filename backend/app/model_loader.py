"""Loads the SER model once (at app startup, see main.py's lifespan
handler) and exposes a predict function. Same bridge-into-ser_pipeline
approach as the Streamlit version, minus the Streamlit-specific caching
(FastAPI just loads it once at process start instead)."""
import os
import sys
import tempfile

from . import settings

_SER_PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ser_pipeline')
if _SER_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _SER_PIPELINE_DIR)

_predict_emotion = None  # populated by load_model() at startup


def _ensure_checkpoint_downloaded(model_path: str):
    if os.path.exists(model_path):
        return
    if not settings.MODEL_URL:
        raise FileNotFoundError(
            f'Model checkpoint not found at {model_path} and no MODEL_URL is set.')
    import urllib.request
    print(f'Downloading model checkpoint from {settings.MODEL_URL} ...')
    tmp_path = model_path + '.part'
    urllib.request.urlretrieve(settings.MODEL_URL, tmp_path)
    os.replace(tmp_path, model_path)
    print('Model checkpoint downloaded.')


def load_model():
    """Call once at startup. Populates the module-level predict function."""
    global _predict_emotion
    if _predict_emotion is not None:
        return  # already loaded

    import config as ser_config  # noqa: local import, path set up above
    _ensure_checkpoint_downloaded(ser_config.MODEL_PATH)

    from inference.predictor import predict_emotion
    _predict_emotion = predict_emotion
    print('SER model loaded and ready.')


def analyze_audio_bytes(audio_bytes: bytes, suffix: str = '.wav') -> dict:
    if _predict_emotion is None:
        raise RuntimeError('Model not loaded yet — load_model() must run at startup.')
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        return _predict_emotion(tmp_path)
    finally:
        os.unlink(tmp_path)
