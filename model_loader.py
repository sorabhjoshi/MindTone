"""Loads the SER model once per app process (Streamlit's @st.cache_resource
keeps it warm across reruns/users) and exposes a single predict function.

The 385MB checkpoint doesn't belong in git. On first run, if it isn't
already on disk, this downloads it from MODEL_URL (set that to a
Hugging Face Hub file URL, or any direct-download link) into the app's
working directory.
"""
import os
import sys
import tempfile

import streamlit as st

# Make the bundled ser_pipeline package's own internal absolute imports
# (e.g. "from data.audio import ...") resolve correctly — it expects to
# be run with itself as the sys.path root, same as train.py/infer.py do.
_SER_PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ser_pipeline')
if _SER_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _SER_PIPELINE_DIR)

# Route the pipeline's WORK_DIR (and therefore MODEL_PATH) to a local
# folder here instead of the Kaggle-default /kaggle/working, BEFORE
# config.py is imported anywhere (imports are cached after first import).
_APP_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_data')
os.makedirs(_APP_DATA_DIR, exist_ok=True)
os.environ.setdefault('SER_WORK_DIR', _APP_DATA_DIR)

MODEL_URL = os.environ.get('MODEL_URL', '')  # e.g. a Hugging Face Hub direct-download URL


def _ensure_checkpoint_downloaded(model_path: str):
    if os.path.exists(model_path):
        return
    if not MODEL_URL:
        raise FileNotFoundError(
            f'Model checkpoint not found at {model_path} and no MODEL_URL is set. '
            'Either place ser_model.pth there manually, or set the MODEL_URL '
            'environment variable to a direct-download link (e.g. a Hugging '
            'Face Hub file URL).')
    import urllib.request
    st_status = st.empty()
    st_status.info('Downloading model checkpoint (first run only, ~385MB) ...')
    tmp_path = model_path + '.part'
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    os.replace(tmp_path, model_path)
    st_status.empty()


@st.cache_resource(show_spinner='Loading model ...')
def load_predictor():
    """Returns predict_emotion, with the checkpoint guaranteed present
    on disk first."""
    import config as ser_config  # noqa: local import, path set up above
    _ensure_checkpoint_downloaded(ser_config.MODEL_PATH)

    from inference.predictor import predict_emotion

    # Touch the model once here so the (potentially slow, first-time HF
    # download of the base backbone) happens during cached load, not on
    # the user's first check-in.
    return predict_emotion


def analyze_audio_bytes(audio_bytes: bytes, suffix: str = '.wav') -> dict:
    """Writes the recorded audio to a temp file (predict_emotion needs a
    real path) and returns the emotion prediction — nothing more. Any
    long-term pattern analysis is computed separately from accumulated
    history (see patterns.py), not derived per-entry here."""
    predict_emotion = load_predictor()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        return predict_emotion(tmp_path)
    finally:
        os.unlink(tmp_path)
