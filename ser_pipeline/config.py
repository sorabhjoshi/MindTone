"""
Central configuration for the Speech Emotion Recognition pipeline.
Locked-in recipe: confirmed test WA=71.46% / UA=70.63% (TTA x3).

Everything a run needs to know lives here. train.py and infer.py both
import from this module instead of hardcoding values.
"""
import os
import random

import numpy as np
import torch

# ── Reproducibility ─────────────────────────────────────────────────────
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset roots ───────────────────────────────────────────────────────
# Overridable via environment variables so this runs unchanged on Kaggle,
# a local machine, or any other host.
RAVDESS_ROOT = os.environ.get(
    'SER_RAVDESS_ROOT',
    '/kaggle/input/datasets/uwrfkaggler/ravdess-emotional-speech-audio')
CREMAD_ROOT = os.environ.get(
    'SER_CREMAD_ROOT', '/kaggle/input/datasets/ejlok1/cremad')
IEMOCAP_ROOT = os.environ.get(
    'SER_IEMOCAP_ROOT',
    '/kaggle/input/datasets/dejolilandry/iemocapfullrelease/IEMOCAP_full_release')

WORK_DIR = os.environ.get('SER_WORK_DIR', '/kaggle/working')
CACHE_DIR = WORK_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Labels ──────────────────────────────────────────────────────────────
LABEL_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
NUM_CLASSES = len(LABEL_NAMES)
L2I = {l: i for i, l in enumerate(LABEL_NAMES)}

DATASET_NAMES = ['RAVDESS', 'CREMA-D', 'IEMOCAP']
NUM_DOMAINS = len(DATASET_NAMES)
DS2I = {d: i for i, d in enumerate(DATASET_NAMES)}

# ── Audio config ────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
MAX_SECONDS = 4.0

# Silence trimming (energy-based, adapts to each clip's own peak loudness)
SIL_THRESH_RATIO = 0.02   # frame RMS below 2% of peak RMS counts as silence
SIL_FRAME_MS = 25
SIL_PAD_MS = 100

# Wav-level augmentation
WAV_NOISE_PROB = 0.40
WAV_NOISE_SNR = (15, 40)      # dB range
WAV_SPEED_PROB = 0.35
WAV_SPEED_RANGE = (0.90, 1.10)  # +/-10% speed

# ── Model config ────────────────────────────────────────────────────────
# LOCKED-IN — this is the confirmed 71.46%/70.63% recipe. See README before
# changing W2V_MODEL/N_W2V_UNFROZEN/DOMAIN_LOSS_WEIGHT: every attempt to
# improve on this (backbone swaps, capacity bumps, extra loss terms,
# larger backbones) regressed. Only change these deliberately, one at a
# time, and expect to beat 71.46% test WA to justify keeping the change.
W2V_MODEL = 'microsoft/wavlm-base-plus'
W2V_DIM = 768                 # hidden size for wavlm-base-plus
N_LAYERS_USE = 4               # last N transformer layers, layer-weighted
N_W2V_UNFROZEN = 4              # last N transformer layers fine-tuned

# ── Training config ─────────────────────────────────────────────────────
BATCH_SIZE = 16
GRAD_ACCUM = 6                 # effective batch = 96
NUM_EPOCHS = 170                # ceiling — PATIENCE stops runs well before this
PATIENCE = 30

LR_W2V_PROJ = 3e-5
LR_HEAD = 2e-4
LR_W2V_FT = 5e-6               # base LR for the layer closest to the head
LR_W2V_LAYER_DECAY = 0.7        # earlier unfrozen layers get LR *= this per layer

WEIGHT_DECAY = 1e-3
LABEL_SMOOTH = 0.05
DROPOUT_HEAD = 0.50
DROPOUT_PROJ = 0.35
FOCAL_GAMMA = 1.5

TTA_N = 3
TEMP_CALIB = True
EMA_DECAY = 0.995

# ── Domain-adversarial config ──────────────────────────────────────────
# DOMAIN_LOSS_WEIGHT=0.0 is a deliberate ablation, not an oversight — with
# 3 domains the adversarial signal was unstable and showed no evidence of
# helping class-level accuracy. Re-enabling it (tried at 0.15) did not beat
# this baseline either. See README for the full experiment log.
DOMAIN_LOSS_WEIGHT = 0.0
GRL_LAMBDA_MAX = 0.5

# ── Pairwise contrastive loss ───────────────────────────────────────────
# Extra loss weight when the model's top-2 logits are exactly Fear vs Happy
# — the single largest confusion pair in the confusion matrix.
CONTRAST_CLASS_A = 'Fear'
CONTRAST_CLASS_B = 'Happy'
CONTRAST_EXTRA_WEIGHT = 1.5

# ── Stage 1 (emotion-adapt the backbone before full training) ──────────
STAGE1_N_W2V_UNFROZEN = 6       # unfreeze more here — no classifier head to
                                 # coordinate with yet, so it's cheap
STAGE1_EPOCHS = 8
STAGE1_LR = 2e-5
STAGE1_WEIGHT_DECAY = 1e-4

# ── Cache / checkpoint paths ────────────────────────────────────────────
CACHE_WAV_TR = os.path.join(CACHE_DIR, 'wav_train_v2.pt')
CACHE_WAV_VA = os.path.join(CACHE_DIR, 'wav_val_v2.pt')
CACHE_WAV_TE = os.path.join(CACHE_DIR, 'wav_test_v2.pt')

# Stage 1 checkpoint is named per-backbone so switching W2V_MODEL across
# runs can never silently load a shape-mismatched checkpoint from a
# different backbone left over in WORK_DIR.
STAGE1_W2V_PATH = os.path.join(
    WORK_DIR, f"w2v_emotion_pretrained_{W2V_MODEL.replace('/', '_')}.pt")

MODEL_PATH = os.path.join(WORK_DIR, 'ser_model.pth')
HISTORY_CSV = os.path.join(WORK_DIR, 'training_history.csv')
CURVES_PNG = os.path.join(WORK_DIR, 'training_curves.png')
CONFUSION_PNG = os.path.join(WORK_DIR, 'confusion_matrix.png')


def print_summary() -> None:
    print(f'Device  : {DEVICE}')
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f'  GPU   : {p.name}  |  VRAM: {p.total_memory / 1e9:.1f} GB')
    print(f'Model   : {W2V_MODEL}  |  dim={W2V_DIM}  |  layers={N_LAYERS_USE}  '
          f'|  unfrozen={N_W2V_UNFROZEN}')
    print(f'Classes : {NUM_CLASSES}  |  {LABEL_NAMES}')
    print(f'Effective batch: {BATCH_SIZE * GRAD_ACCUM}')
    print(f'Domain loss weight: {DOMAIN_LOSS_WEIGHT}')
