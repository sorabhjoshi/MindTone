"""Standalone inference: rebuild a SERModel purely from a saved checkpoint
and run it on a single audio file. No dependency on any in-memory
training state — this is what makes a checkpoint self-contained."""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import config
from data.audio import augment_wav, load_wav
from engine.amp import make_autocast
from model.ser_model import SERModel

_inference_model_cache: dict = {}


def load_model_for_inference(model_path: str = None, device=None):
    """Cached by path so repeated calls in the same process don't reload
    from disk and rebuild the backbone every time."""
    model_path = model_path or config.MODEL_PATH
    device = device or config.DEVICE
    if model_path in _inference_model_cache:
        return _inference_model_cache[model_path]

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    m = SERModel(
        num_classes=ckpt['num_classes'],
        n_w2v_unfrozen=cfg['n_w2v_unfrozen'],
        w2v_model=cfg['w2v_model'],
        w2v_dim=cfg['w2v_dim'],
        n_layers_use=cfg['n_layers_use'],
        load_stage1=False,   # loading the final trained state_dict next anyway
    ).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()

    bundle = {
        'model': m,
        'temperature': float(ckpt['temperature']),
        'vector_scale': ckpt['vector_scale'],
        'label_names': ckpt['label_names'],
    }
    _inference_model_cache[model_path] = bundle
    return bundle


@torch.no_grad()
def predict_emotion(audio_path: str, ema_model=None, n_tta: int = config.TTA_N,
                     model_path: str = None, temperature: float = None,
                     vector_scale: torch.Tensor = None, device=config.DEVICE):
    """ema_model lets you pass a live training-session EMA shadow directly
    without a disk round-trip. Leave it None to load purely from
    model_path/config.MODEL_PATH on disk — the path a fresh process uses."""
    if ema_model is not None:
        infer_model = ema_model
        infer_model.eval()
        infer_T = temperature if temperature is not None else 1.0
        infer_vs = vector_scale if vector_scale is not None else torch.ones(config.NUM_CLASSES)
    else:
        bundle = load_model_for_inference(model_path)
        infer_model = bundle['model']
        infer_T = bundle['temperature']
        infer_vs = bundle['vector_scale']

    wav, wav_len = load_wav(audio_path)
    wln_t = torch.tensor([wav_len], dtype=torch.long).to(device)

    def _infer_once(aug: bool):
        w = wav.clone()
        if aug and random.random() < 0.30:
            w = augment_wav(w, noise_prob=1.0, speed_prob=0.0)
        w = w.unsqueeze(0).to(device)
        with make_autocast():
            raw = infer_model(w, wav_len=wln_t)
        return (raw.float() / infer_T) * infer_vs.to(raw.device)

    logits_clean = _infer_once(aug=False)
    if n_tta > 0:
        aug_sum = sum(_infer_once(aug=True) for _ in range(n_tta))
        final_logits = 0.5 * logits_clean + 0.5 * (aug_sum / n_tta)
    else:
        final_logits = logits_clean

    probs = F.softmax(final_logits, dim=1).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)) / np.log(config.NUM_CLASSES))

    return {
        'predicted_emotion': config.LABEL_NAMES[pred],
        'confidence': float(probs[pred]),
        'uncertainty': round(entropy, 4),
        'class_probabilities': {n: round(float(p), 4)
                                 for n, p in zip(config.LABEL_NAMES, probs)},
        'audio_file': os.path.basename(audio_path),
    }
