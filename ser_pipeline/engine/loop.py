"""One training epoch, plain evaluation, and TTA evaluation."""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from data.dataset import SERDataset
from model.layers import grl_lambda_schedule
from engine.amp import make_autocast


def train_epoch(model, ema, loader, opt, sched, crit, contrast_loss, scaler,
                 epoch: int, n_epochs: int, device=config.DEVICE,
                 domain_loss_weight: float = config.DOMAIN_LOSS_WEIGHT,
                 grad_accum: int = config.GRAD_ACCUM):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    dom_preds, dom_labels = [], []
    grl_lambda = grl_lambda_schedule(epoch, n_epochs)
    opt.zero_grad(set_to_none=True)

    for step, (wav, wav_len, y, dom) in enumerate(
            tqdm(loader, desc='Train', leave=True, file=sys.stdout)):
        wav, wav_len, y, dom = (wav.to(device, non_blocking=True),
                                 wav_len.to(device, non_blocking=True),
                                 y.to(device, non_blocking=True),
                                 dom.to(device, non_blocking=True))
        with make_autocast():
            logits, dom_logits = model(wav, wav_len=wav_len,
                                        return_domain=True, grl_lambda=grl_lambda)
            per_sample_loss = crit(logits, y)
            emo_loss = contrast_loss(logits, y, per_sample_loss)
            dom_loss = F.cross_entropy(dom_logits, dom)
            loss = (emo_loss + domain_loss_weight * dom_loss) / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            ema.update(model)
        total_loss += loss.item() * grad_accum * len(y)
        with torch.no_grad():
            all_preds.extend(logits.argmax(1).cpu().numpy())
            dom_preds.extend(dom_logits.argmax(1).cpu().numpy())
            dom_labels.extend(dom.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n = len(all_labels)
    dom_preds_arr, dom_labels_arr = np.array(dom_preds), np.array(dom_labels)
    dom_acc = accuracy_score(dom_labels_arr, dom_preds_arr)
    dom_bal_acc = balanced_accuracy_score(dom_labels_arr, dom_preds_arr)
    dom_majority = max(np.bincount(
        dom_labels_arr, minlength=config.NUM_DOMAINS)) / max(1, n)
    # dom_bal_acc near 0.5 means the GRL is doing its job. dom_acc alone can
    # look artificially low/high purely from imbalance, hence dom_majority.
    print(f'    [domain] grl_lambda={grl_lambda:.3f}  domain_acc={dom_acc:.3f}  '
          f'domain_balanced_acc={dom_bal_acc:.3f}  majority_baseline={dom_majority:.3f}',
          file=sys.stdout)
    return (total_loss / n, accuracy_score(all_labels, all_preds),
            balanced_accuracy_score(all_labels, all_preds))


@torch.no_grad()
def evaluate(ema_model, loader, crit, temperature: float = 1.0, device=config.DEVICE):
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for wav, wav_len, y, dom in loader:
        wav, wav_len, y = (wav.to(device, non_blocking=True),
                            wav_len.to(device, non_blocking=True),
                            y.to(device, non_blocking=True))
        with make_autocast():
            logits = ema_model(wav, wav_len=wav_len) / temperature
        loss = crit(logits.float(), y).mean()
        total_loss += loss.item() * len(y)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(probs.argmax(1))
        all_labels.extend(y.cpu().numpy())
    n = len(all_labels)
    return (total_loss / n,
            accuracy_score(all_labels, all_preds),
            balanced_accuracy_score(all_labels, all_preds),
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


@torch.no_grad()
def evaluate_tta(ema_model, wav_cache, labels_list, crit, temperature: float = 1.0,
                  n_tta: int = config.TTA_N, wav_lens=None, device=config.DEVICE,
                  batch_size: int = config.BATCH_SIZE):
    def _get_logits(augment: bool):
        ds = SERDataset(wav_cache, labels_list, domains=[0] * len(labels_list),
                         augment=augment, wav_lens=wav_lens)
        ld = DataLoader(ds, batch_size=batch_size * 2, shuffle=False,
                         num_workers=2, pin_memory=True, persistent_workers=True)
        out = []
        for wav, wln, _, _ in ld:
            wav, wln = wav.to(device, non_blocking=True), wln.to(device, non_blocking=True)
            with make_autocast():
                out.append(ema_model(wav, wav_len=wln).float().cpu())
        return torch.cat(out, dim=0) / temperature

    logits_clean = _get_logits(augment=False)
    logits_aug = sum(_get_logits(augment=True) for _ in range(n_tta)) / n_tta
    final_logits = 0.5 * logits_clean + 0.5 * logits_aug
    probs = F.softmax(final_logits, dim=1).numpy()
    preds = probs.argmax(1)
    labels = np.array(labels_list)
    return (accuracy_score(labels, preds),
            balanced_accuracy_score(labels, preds),
            preds, labels, probs)
