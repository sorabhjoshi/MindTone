"""The final model: a fine-tuned wav2vec2/WavLM backbone (last N layers
unfrozen) -> layer-weighted sum -> temporal attention pooling -> a small
classifier head, plus a domain-adversarial head behind a gradient
reversal layer. No CNN/mel branch — matches the confirmed 71.46%/70.63%
recipe (see README)."""
import os

import torch
import torch.nn as nn

import config
from model.layers import (LayerWeightedSum, TemporalAttentionPooling,
                           _load_w2v_backbone, grad_reverse)


class SERModel(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 dropout_head: float = config.DROPOUT_HEAD,
                 dropout_proj: float = config.DROPOUT_PROJ,
                 n_w2v_unfrozen: int = config.N_W2V_UNFROZEN,
                 w2v_model: str = config.W2V_MODEL,
                 w2v_dim: int = config.W2V_DIM,
                 n_layers_use: int = config.N_LAYERS_USE,
                 num_domains: int = config.NUM_DOMAINS,
                 load_stage1: bool = True):
        super().__init__()
        self.n_layers_use = n_layers_use

        # ── backbone, last N transformer layers unfrozen ────────────────
        self.w2v_backbone = _load_w2v_backbone(w2v_model)

        if load_stage1:
            stage1_path = os.path.join(
                config.WORK_DIR, f"w2v_emotion_pretrained_{w2v_model.replace('/', '_')}.pt")
            if os.path.exists(stage1_path):
                try:
                    self.w2v_backbone.load_state_dict(
                        torch.load(stage1_path, map_location='cpu'))
                    print('  Loaded Stage 1 emotion-adapted backbone weights.')
                except RuntimeError as e:
                    # A stale checkpoint from a different backbone
                    # shouldn't crash the run.
                    print(f'  Stage 1 checkpoint at {stage1_path} does not match '
                          f'this backbone, skipping it:\n    {e}')

        for p in self.w2v_backbone.parameters():
            p.requires_grad = False
        n_layers_total = len(self.w2v_backbone.encoder.layers)
        for layer in self.w2v_backbone.encoder.layers[n_layers_total - n_w2v_unfrozen:]:
            for p in layer.parameters():
                p.requires_grad = True
        # feature_extractor (the CNN front-end) stays frozen — raw
        # waveform-to-frame conversion is stable and doesn't need adapting.

        self.layer_weight = LayerWeightedSum(n_layers=n_layers_use)
        self.attn_pool = TemporalAttentionPooling(dim=w2v_dim)
        w2v_out = 512
        self.w2v_proj = nn.Sequential(
            nn.Linear(w2v_dim, 768), nn.GELU(), nn.LayerNorm(768),
            nn.Dropout(dropout_proj),
            nn.Linear(768, w2v_out), nn.LayerNorm(w2v_out))

        self.head = nn.Sequential(
            nn.Dropout(dropout_head),
            nn.Linear(w2v_out, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_head * 0.4),
            nn.Linear(256, num_classes))

        # domain classifier, sits behind the GRL
        self.domain_head = nn.Sequential(
            nn.Linear(w2v_out, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_domains))

    def forward(self, wav, wav_len=None, return_domain: bool = False,
                grl_lambda: float = 1.0):
        raw_mask = None
        if wav_len is not None:
            T = wav.shape[1]
            raw_mask = (torch.arange(T, device=wav.device).unsqueeze(0)
                        < wav_len.unsqueeze(1)).long()          # (B, T_raw)
        w2v_out = self.w2v_backbone(wav, attention_mask=raw_mask,
                                     output_hidden_states=True)
        last_layers = torch.stack(
            w2v_out.hidden_states[-self.n_layers_use:], dim=1)  # (B, L, T, D)
        frame_repr = self.layer_weight(last_layers)              # (B, T, D)

        feat_mask = None
        if raw_mask is not None:
            feat_mask = self.w2v_backbone._get_feature_vector_attention_mask(
                frame_repr.shape[1], raw_mask)
        w_pooled = self.attn_pool(frame_repr, mask=feat_mask)    # (B, D)
        w = self.w2v_proj(w_pooled)

        emotion_logits = self.head(w)

        if return_domain:
            domain_logits = self.domain_head(grad_reverse(w, grl_lambda))
            return emotion_logits, domain_logits
        return emotion_logits
