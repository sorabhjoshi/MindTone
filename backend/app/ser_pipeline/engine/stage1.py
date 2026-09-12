"""Stage 1: pretrain the backbone's unfrozen layers on emotion
classification alone, before the full model (with domain-adversarial
head etc.) trains on top of it. Mirrors P-TAPT-style task adaptation:
adapt the speech backbone to emotion first, rather than fine-tuning
everything simultaneously from a cold start."""
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from data.dataset import SERDataset
from model.layers import LayerWeightedSum, TemporalAttentionPooling, _load_w2v_backbone
from engine.amp import make_autocast, make_scaler


class W2VEmotionProbe(nn.Module):
    """Lightweight temporary head — just enough to force the backbone to
    specialize toward emotion. Uses the same pooling as the final
    SERModel so Stage 1's adapted weights are optimized under the same
    pooling behavior they'll actually be used with later."""
    def __init__(self, w2v_model: str = config.W2V_MODEL,
                 n_unfrozen: int = config.STAGE1_N_W2V_UNFROZEN,
                 num_classes: int = config.NUM_CLASSES,
                 w2v_dim: int = config.W2V_DIM,
                 n_layers_use: int = config.N_LAYERS_USE):
        super().__init__()
        self.n_layers_use = n_layers_use
        self.backbone = _load_w2v_backbone(w2v_model)
        for p in self.backbone.parameters():
            p.requires_grad = False
        n_total = len(self.backbone.encoder.layers)
        for layer in self.backbone.encoder.layers[n_total - n_unfrozen:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.layer_weight = LayerWeightedSum(n_layers=n_layers_use)
        self.pool = TemporalAttentionPooling(dim=w2v_dim)
        self.head = nn.Sequential(
            nn.Linear(w2v_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes))

    def forward(self, wav, wav_len=None):
        raw_mask = None
        if wav_len is not None:
            T = wav.shape[1]
            raw_mask = (torch.arange(T, device=wav.device).unsqueeze(0)
                        < wav_len.unsqueeze(1)).long()
        out = self.backbone(wav, attention_mask=raw_mask, output_hidden_states=True)
        layer_stack = torch.stack(out.hidden_states[-self.n_layers_use:], dim=1)
        frame_repr = self.layer_weight(layer_stack)
        feat_mask = None
        if raw_mask is not None:
            feat_mask = self.backbone._get_feature_vector_attention_mask(
                frame_repr.shape[1], raw_mask)
        pooled = self.pool(frame_repr, mask=feat_mask)
        return self.head(pooled)


def run_stage1_pretraining(wav_train, labels_train, domains_train, wav_lens_train,
                            sampler, device=config.DEVICE,
                            w2v_model: str = config.W2V_MODEL,
                            epochs: int = config.STAGE1_EPOCHS,
                            lr: float = config.STAGE1_LR,
                            weight_decay: float = config.STAGE1_WEIGHT_DECAY,
                            batch_size: int = config.BATCH_SIZE) -> str:
    """Trains the probe and saves the emotion-adapted backbone weights to
    a path named per-backbone (see config.STAGE1_W2V_PATH), so switching
    W2V_MODEL across runs can never load a shape-mismatched checkpoint.
    Returns the save path."""
    print('Stage 1: pretraining backbone on emotion classification only ...')
    probe = W2VEmotionProbe(w2v_model=w2v_model).to(device)
    probe_opt = torch.optim.AdamW(
        [p for p in probe.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    probe_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    probe_scaler = make_scaler()

    ds = SERDataset(wav_train, labels_train, domains_train, augment=True,
                     wav_lens=wav_lens_train)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                         num_workers=2, pin_memory=True, drop_last=True,
                         persistent_workers=True)

    probe.train()
    for epoch in range(1, epochs + 1):
        total, correct, total_loss = 0, 0, 0.0
        for wav, wav_len, y, dom in tqdm(loader, desc=f'Stage1 ep{epoch}', file=sys.stdout):
            wav, wav_len, y = (wav.to(device, non_blocking=True),
                                wav_len.to(device, non_blocking=True),
                                y.to(device, non_blocking=True))
            probe_opt.zero_grad(set_to_none=True)
            with make_autocast():
                logits = probe(wav, wav_len=wav_len)
                loss = probe_crit(logits, y)
            probe_scaler.scale(loss).backward()
            probe_scaler.step(probe_opt)
            probe_scaler.update()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        print(f'  Stage1 epoch {epoch}: loss={total_loss / total:.4f}  acc={correct / total:.4f}')

    save_path = config.STAGE1_W2V_PATH
    torch.save(probe.backbone.state_dict(), save_path)
    print(f'Stage 1 complete. Emotion-adapted backbone saved to {save_path}')
    del probe, probe_opt, ds, loader
    torch.cuda.empty_cache()
    return save_path
