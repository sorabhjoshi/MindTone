#!/usr/bin/env python3
"""
Full training pipeline: load data -> speaker split -> build wav cache ->
Stage 1 backbone pretraining -> main training loop -> calibration ->
final test evaluation -> save checkpoint + plots.

Usage:
    python train.py                      # run with config.py defaults
    python train.py --skip-stage1        # skip Stage 1 pretraining
    python train.py --epochs 50          # override epoch ceiling

Environment variables (see config.py) let you point this at datasets
and a working directory outside of Kaggle:
    SER_RAVDESS_ROOT, SER_CREMAD_ROOT, SER_IEMOCAP_ROOT, SER_WORK_DIR
"""
import argparse
import copy
import os

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

import config
from data.loaders import load_all_records
from data.split import make_splits
from data.cache import build_wav_cache
from data.dataset import SERDataset
from engine.amp import make_scaler
from engine.calibration import collect_val_logits, fit_calibration, make_calibrate_fn
from engine.ema import ModelEMA
from engine.loop import evaluate, evaluate_tta, train_epoch
from engine.optim import build_optimizer, build_scheduler
from engine.stage1 import run_stage1_pretraining
from model.layers import FocalLoss, PairwiseContrastLoss
from model.ser_model import SERModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    p.add_argument('--patience', type=int, default=config.PATIENCE)
    p.add_argument('--skip-stage1', action='store_true',
                    help='Skip Stage 1 backbone pretraining (uses stock pretrained '
                         'backbone weights instead of emotion-adapted ones).')
    p.add_argument('--no-plots', action='store_true',
                    help='Skip matplotlib training-curve / confusion-matrix plots '
                         '(useful on headless machines without a display backend).')
    return p.parse_args()


def gpu_alive(ctx: str = ''):
    if not torch.cuda.is_available():
        return  # CPU runs are allowed (slow), just no GPU-liveness check to do
    try:
        torch.zeros(1, device='cuda')
    except Exception as e:
        raise RuntimeError(f'GPU unresponsive{" during " + ctx if ctx else ""}: {e}') from e


def main():
    args = parse_args()
    config.seed_everything()
    config.print_summary()

    # ── Data ──────────────────────────────────────────────────────────
    df = load_all_records()
    df_train, df_val, df_test, class_weights, sampler = make_splits(df)

    labels_tr = df_train['emotion'].tolist()
    labels_va = df_val['emotion'].tolist()
    labels_te = df_test['emotion'].tolist()
    domains_tr = df_train['domain'].tolist()
    domains_va = df_val['domain'].tolist()

    wav_train, wav_lens_train = build_wav_cache(df_train, config.CACHE_WAV_TR, 'train')
    wav_val, wav_lens_val = build_wav_cache(df_val, config.CACHE_WAV_VA, 'val')
    wav_test, wav_lens_test = build_wav_cache(df_test, config.CACHE_WAV_TE, 'test')

    # ── Stage 1: emotion-adapt the backbone ──────────────────────────
    if not args.skip_stage1:
        run_stage1_pretraining(wav_train, labels_tr, domains_tr, wav_lens_train, sampler)
    else:
        print('Skipping Stage 1 (--skip-stage1) — backbone starts from stock pretrained weights.')

    # ── Dataloaders ───────────────────────────────────────────────────
    gpu_alive('before training setup')
    tr_ds = SERDataset(wav_train, labels_tr, domains_tr, augment=True, wav_lens=wav_lens_train)
    va_ds = SERDataset(wav_val, labels_va, domains_va, augment=False, wav_lens=wav_lens_val)
    tr_ldr = DataLoader(tr_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
                         num_workers=2, pin_memory=True, drop_last=True,
                         persistent_workers=True)
    va_ldr = DataLoader(va_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                         num_workers=2, pin_memory=True, persistent_workers=True)

    # ── Model, optimizer, loss ────────────────────────────────────────
    model = SERModel().to(config.DEVICE)
    ema = ModelEMA(model, decay=config.EMA_DECAY)
    # Imbalance correction lives in ONE place: WeightedRandomSampler already
    # oversamples minority classes at the batch level. Passing class weights
    # into the loss too would stack both corrections and overcorrect.
    criterion = FocalLoss(gamma=config.FOCAL_GAMMA, label_smoothing=config.LABEL_SMOOTH)
    contrast_loss = PairwiseContrastLoss(
        config.L2I[config.CONTRAST_CLASS_A], config.L2I[config.CONTRAST_CLASS_B],
        extra_weight=config.CONTRAST_EXTRA_WEIGHT)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, len(tr_ldr) // config.GRAD_ACCUM, args.epochs)
    scaler = make_scaler()

    history = []
    best_ua, best_wa, best_ema_state, no_improve = 0.0, 0.0, None, 0

    print('=' * 75, flush=True)
    print('  Speech Emotion Recognition — fine-tuned backbone + domain-adversarial head',
          flush=True)
    print(f'  train={len(df_train)} ({df_train["speaker"].nunique()} spk)  '
          f'val={len(df_val)} ({df_val["speaker"].nunique()} spk)  '
          f'test={len(df_test)} ({df_test["speaker"].nunique()} spk)', flush=True)
    print(f'  Effective batch={config.BATCH_SIZE * config.GRAD_ACCUM}  |  EMA decay={config.EMA_DECAY}',
          flush=True)
    print('=' * 75, flush=True)
    print(f"  {'Ep':>4}  {'tr_WA':>7}  {'val_WA':>7}  {'val_UA':>7}  {'gap':>6}  status", flush=True)
    print('-' * 75, flush=True)

    for epoch in range(1, args.epochs + 1):
        if epoch % 20 == 1:
            gpu_alive(f'epoch {epoch}')

        tr_loss, tr_wa, tr_ua = train_epoch(
            model, ema, tr_ldr, optimizer, scheduler, criterion, contrast_loss,
            scaler, epoch, args.epochs)
        va_loss, va_wa, va_ua, _, _, _ = evaluate(ema.shadow, va_ldr, criterion)

        history.append({'epoch': epoch, 'tr_loss': tr_loss, 'tr_wa': tr_wa, 'tr_ua': tr_ua,
                         'va_loss': va_loss, 'va_wa': va_wa, 'va_ua': va_ua})

        if va_ua > best_ua + 1e-4:
            best_ua, best_wa = va_ua, va_wa
            best_ema_state = copy.deepcopy(ema.shadow.state_dict())
            no_improve, tag = 0, '<- BEST (EMA)'
        else:
            no_improve += 1
            tag = f'(no improve {no_improve}/{args.patience})'

        gap = tr_wa - va_wa
        flag = '  *** OVERFIT ***' if gap > 0.20 else ''
        print(f"  {epoch:4d}  {tr_wa:7.3f}  {va_wa:7.3f}  {va_ua:7.3f}  {gap:+6.3f}  {tag}{flag}",
              flush=True)

        if epoch % 10 == 0 and best_ema_state is not None:
            torch.save({'model_state_dict': best_ema_state, 'epoch': epoch, 'best_ua': best_ua},
                       os.path.join(config.WORK_DIR, f'ckpt_ep{epoch}.pth'))

        if no_improve >= args.patience:
            print(f'\n  Early stop at epoch {epoch}  (best val UA={best_ua:.4f})', flush=True)
            break

    print(f'\nTraining complete. Best EMA val WA={best_wa*100:.2f}%  UA={best_ua*100:.2f}%', flush=True)

    # ── Calibration ───────────────────────────────────────────────────
    ema.shadow.load_state_dict(best_ema_state)
    ema.shadow.eval()
    logits_val, labels_val = collect_val_logits(ema.shadow, va_ldr)
    temperature, vector_scale = fit_calibration(logits_val, labels_val)
    calibrate = make_calibrate_fn(temperature, vector_scale)

    # ── Final test evaluation ────────────────────────────────────────
    @torch.no_grad()
    def evaluate_calibrated(ema_model, loader):
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from engine.amp import make_autocast
        import numpy as np
        total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
        for wav, wav_len, y, dom in loader:
            wav, wav_len, y = (wav.to(config.DEVICE, non_blocking=True),
                                wav_len.to(config.DEVICE, non_blocking=True),
                                y.to(config.DEVICE, non_blocking=True))
            with make_autocast():
                logits = calibrate(ema_model(wav, wav_len=wav_len).float())
            loss = criterion(logits, y).mean()
            total_loss += loss.item() * len(y)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(probs.argmax(1))
            all_labels.extend(y.cpu().numpy())
        n = len(all_labels)
        return (total_loss / n, accuracy_score(all_labels, all_preds),
                balanced_accuracy_score(all_labels, all_preds),
                np.array(all_preds), np.array(all_labels), np.array(all_probs))

    te_ldr = DataLoader(
        SERDataset(wav_test, labels_te, domains=[0] * len(labels_te), augment=False,
                   wav_lens=wav_lens_test),
        batch_size=config.BATCH_SIZE * 2, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True)

    te_loss, te_wa, te_ua, preds_clean, labels_test, probs_clean = evaluate_calibrated(
        ema.shadow, te_ldr)

    print('Running batched TTA on test set ...')
    te_wa_tta, te_ua_tta, preds_tta, labels_tta, probs_tta = evaluate_tta(
        ema.shadow, wav_test, labels_te, criterion, temperature=temperature,
        wav_lens=wav_lens_test)

    print('=' * 65)
    print(f'  Test set ({len(df_test)} samples | {df_test["speaker"].nunique()} speakers)')
    print(f'  Clean  : WA={te_wa*100:.2f}%   UA={te_ua*100:.2f}%')
    print(f'  TTA x{config.TTA_N}: WA={te_wa_tta*100:.2f}%   UA={te_ua_tta*100:.2f}%')
    print('=' * 65)
    print('\nClassification Report (TTA):')
    print(classification_report(labels_test, preds_tta, target_names=config.LABEL_NAMES, digits=3))

    cm = confusion_matrix(labels_test, preds_tta, labels=list(range(config.NUM_CLASSES)))
    print('Confusion matrix (rows=true, cols=predicted):')
    header = '           ' + ''.join(f'{n:>10s}' for n in config.LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f'{config.LABEL_NAMES[i]:>10s} ' + ''.join(f'{v:>10d}' for v in row))

    # ── Save ──────────────────────────────────────────────────────────
    torch.save({
        'model_state_dict': best_ema_state,
        'temperature': temperature,
        'vector_scale': vector_scale,
        'label_names': config.LABEL_NAMES,
        'num_classes': config.NUM_CLASSES,
        'config': {
            'w2v_model': config.W2V_MODEL,
            'w2v_dim': config.W2V_DIM,
            'n_layers_use': config.N_LAYERS_USE,
            'n_w2v_unfrozen': config.N_W2V_UNFROZEN,
            'sample_rate': config.SAMPLE_RATE,
            'max_seconds': config.MAX_SECONDS,
        },
        'results': {
            'test_wa_clean': float(te_wa), 'test_ua_clean': float(te_ua),
            'test_wa_tta': float(te_wa_tta), 'test_ua_tta': float(te_ua_tta),
        },
    }, config.MODEL_PATH)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(config.HISTORY_CSV, index=False)
    print(f'\nModel saved : {config.MODEL_PATH}')
    print(f'Final       : WA={te_wa_tta*100:.2f}%  UA={te_ua_tta*100:.2f}%  (TTA x{config.TTA_N})')

    # ── Plots ─────────────────────────────────────────────────────────
    if not args.no_plots:
        try:
            plot_results(hist_df, cm, te_wa_tta, te_ua_tta)
        except Exception as e:
            print(f'Plotting failed (non-fatal): {e}')


def plot_results(hist_df: pd.DataFrame, cm, te_wa_tta: float, te_ua_tta: float):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Curves', fontsize=12, fontweight='bold')
    axes[0].plot(hist_df['epoch'], hist_df['tr_loss'], label='train', color='steelblue')
    axes[0].plot(hist_df['epoch'], hist_df['va_loss'], label='val (EMA)', color='tomato')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(hist_df['epoch'], hist_df['tr_wa'], label='train WA', color='steelblue')
    axes[1].plot(hist_df['epoch'], hist_df['va_wa'], label='val WA (EMA)', color='tomato')
    axes[1].axhline(te_wa_tta, color='green', ls='--', lw=1.5, label=f'test TTA={te_wa_tta:.3f}')
    axes[1].set_title('Weighted Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(hist_df['epoch'], hist_df['tr_ua'], label='train UA', color='steelblue')
    axes[2].plot(hist_df['epoch'], hist_df['va_ua'], label='val UA (EMA)', color='tomato')
    axes[2].axhline(te_ua_tta, color='green', ls='--', lw=1.5, label=f'test TTA={te_ua_tta:.3f}')
    axes[2].set_title('Unweighted Accuracy'); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.CURVES_PNG, dpi=120, bbox_inches='tight')
    plt.close(fig)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(f'Test Confusion Matrices (TTA x{config.TTA_N} | EMA)',
                  fontsize=12, fontweight='bold')
    for ax, data, title, fmt in zip(
            ax2, [cm, cm_norm], ['Counts', 'Row-normalised (Recall)'], ['d', '.2f']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=config.LABEL_NAMES, yticklabels=config.LABEL_NAMES)
        ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(config.CONFUSION_PNG, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f'Plots saved : {config.CURVES_PNG}, {config.CONFUSION_PNG}')


if __name__ == '__main__':
    main()
