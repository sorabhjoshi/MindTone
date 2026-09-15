"""Optimizer (with discriminative/layer-wise learning-rate decay across
the unfrozen backbone layers) and LR scheduler."""
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import config


def build_optimizer(model, lr_w2v_ft: float = config.LR_W2V_FT,
                     lr_w2v_layer_decay: float = config.LR_W2V_LAYER_DECAY,
                     lr_w2v_proj: float = config.LR_W2V_PROJ,
                     lr_head: float = config.LR_HEAD,
                     weight_decay: float = config.WEIGHT_DECAY):
    """Each unfrozen backbone layer further from the output gets
    LR *= lr_w2v_layer_decay per step away — layers closer to the
    original pretrained input move less than layers closer to the task
    head (standard discriminative fine-tuning)."""
    w2v_groups = []
    unfrozen_layers = [l for l in model.w2v_backbone.encoder.layers
                        if any(p.requires_grad for p in l.parameters())]
    for depth_from_output, layer in enumerate(reversed(unfrozen_layers)):
        layer_lr = lr_w2v_ft * (lr_w2v_layer_decay ** depth_from_output)
        w2v_groups.append({'params': list(layer.parameters()),
                            'lr': layer_lr, 'weight_decay': weight_decay})

    # Guard against silently dropping any unfrozen backbone param that
    # isn't inside encoder.layers (shouldn't normally happen).
    layer_param_ids = {id(p) for l in unfrozen_layers for p in l.parameters()}
    leftover = [p for p in model.w2v_backbone.parameters()
                if p.requires_grad and id(p) not in layer_param_ids]
    if leftover:
        w2v_groups.append({'params': leftover, 'lr': lr_w2v_ft,
                            'weight_decay': weight_decay})

    return torch.optim.AdamW([
        *w2v_groups,
        {'params': (list(model.layer_weight.parameters()) +
                    list(model.attn_pool.parameters()) +
                    list(model.w2v_proj.parameters())),
         'lr': lr_w2v_proj, 'weight_decay': weight_decay},
        {'params': list(model.head.parameters()),
         'lr': lr_head, 'weight_decay': weight_decay},
    ])


def build_scheduler(opt, steps_per_epoch: int, n_epochs: int):
    """Single-cycle cosine decay over a realistic horizon (past the
    early-stopping PATIENCE, so LR isn't still near-peak when training
    would actually stop)."""
    horizon_epochs = max(60, n_epochs)
    t_max = horizon_epochs * steps_per_epoch
    return CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-7)
