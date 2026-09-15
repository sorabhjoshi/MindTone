"""Post-hoc calibration on the validation set: a global temperature
found by grid search, then per-class vector scaling fit with LBFGS."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import config
from engine.amp import make_autocast


@torch.no_grad()
def collect_val_logits(ema_model, val_loader, device=config.DEVICE):
    all_logits, all_labels = [], []
    for wav, wav_len, y, dom in val_loader:
        wav, wav_len = wav.to(device, non_blocking=True), wav_len.to(device, non_blocking=True)
        with make_autocast():
            all_logits.append(ema_model(wav, wav_len=wav_len).float().cpu())
        all_labels.extend(y.numpy())
    return torch.cat(all_logits, dim=0), torch.tensor(all_labels, dtype=torch.long)


def fit_calibration(logits_val: torch.Tensor, labels_val: torch.Tensor,
                     num_classes: int = config.NUM_CLASSES,
                     enabled: bool = config.TEMP_CALIB):
    """Returns (temperature: float, vector_scale: Tensor[num_classes])."""
    temperature = 1.0
    vector_scale = torch.ones(num_classes)
    if not enabled:
        return temperature, vector_scale

    print('Step 1: Global temperature search ...')
    best_nll, best_T = float('inf'), 1.0
    for T in np.linspace(0.5, 3.0, 101):
        nll = F.cross_entropy(logits_val / T, labels_val).item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    temperature = float(best_T)
    print(f'  T* = {temperature:.3f}  NLL={best_nll:.4f}')

    print('Step 2: Per-class vector scaling (LBFGS) ...')
    scale_vec = nn.Parameter(torch.ones(num_classes))
    opt_vs = torch.optim.LBFGS([scale_vec], lr=0.05, max_iter=200,
                                line_search_fn='strong_wolfe')

    def vs_closure():
        opt_vs.zero_grad()
        loss = F.cross_entropy(logits_val * scale_vec.unsqueeze(0), labels_val)
        loss.backward()
        return loss

    opt_vs.step(vs_closure)
    with torch.no_grad():
        # LBFGS has no positivity constraint; a negative scale would flip
        # that class's logit ranking, never a sane calibration outcome.
        scale_vec.clamp_(min=0.0)
        final_nll = F.cross_entropy(
            logits_val * scale_vec.unsqueeze(0), labels_val).item()
        final_acc = accuracy_score(
            labels_val.numpy(),
            (logits_val * scale_vec.unsqueeze(0)).argmax(1).numpy())
    vector_scale = scale_vec.detach().clone()
    print(f'  Vec-scale: {vector_scale.numpy().round(3)}')
    print(f'  Post-calib NLL={final_nll:.4f}  val_WA={final_acc:.4f}')
    return temperature, vector_scale


def make_calibrate_fn(temperature: float, vector_scale: torch.Tensor):
    def calibrate(logits_t: torch.Tensor) -> torch.Tensor:
        return (logits_t / temperature) * vector_scale.to(logits_t.device)
    return calibrate
