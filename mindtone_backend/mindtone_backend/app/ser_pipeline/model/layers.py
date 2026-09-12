"""Building blocks shared by the Stage 1 probe and the final SERModel."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, WavLMModel

import config


def _load_w2v_backbone(model_name: str):
    """Picks the right HF model class for the checkpoint. WavLM and
    wav2vec2 expose the same API surface used everywhere else here
    (forward(..., output_hidden_states=True) -> .hidden_states,
    .encoder.layers, ._get_feature_vector_attention_mask), so swapping
    W2V_MODEL between the two families needs no other code changes as
    long as the right class is instantiated here."""
    if 'wavlm' in model_name.lower():
        return WavLMModel.from_pretrained(model_name)
    return Wav2Vec2Model.from_pretrained(model_name)


class GradReverse(torch.autograd.Function):
    """Identity on the forward pass, negates (and scales) the gradient on
    the backward pass — the entire trick behind domain-adversarial
    training. A domain classifier sits behind this and is trained
    normally, but because its gradient is negated before reaching the
    shared trunk, the trunk is pushed to make domain harder to tell."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return GradReverse.apply(x, lambd)


def grl_lambda_schedule(epoch: int, n_epochs: int,
                         lambda_max: float = config.GRL_LAMBDA_MAX) -> float:
    """Ramps the reversal strength 0 -> lambda_max over training (standard
    DANN schedule) so the emotion task stabilizes before adversarial
    pressure ramps up."""
    p = epoch / max(1, n_epochs)
    return lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


class LayerWeightedSum(nn.Module):
    """Combines the last N transformer layers with a learnable per-layer
    softmax-normalized weight. Happens per time frame, so no temporal
    information is discarded — output is still (B, T, D)."""
    def __init__(self, n_layers: int = config.N_LAYERS_USE):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layer_stack):
        # layer_stack: (B, L, T, D)
        w = F.softmax(self.raw_weights, dim=0)
        return (layer_stack * w.view(1, -1, 1, 1)).sum(dim=1)   # (B, T, D)


class TemporalAttentionPooling(nn.Module):
    """Attends over time frames so an emotionally loud moment (a pitch
    spike, a shout) can be weighted higher than a quiet stretch of the
    same utterance."""
    def __init__(self, dim: int = config.W2V_DIM):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim))
        self.proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, x, mask=None):
        # x: (B, T, D). mask: optional (B, T), 1 = real frame, 0 = padding.
        keys = self.proj(x)
        scores = (keys * self.query.unsqueeze(0)).sum(-1)       # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores * self.scale, dim=1)
        return (x * weights.unsqueeze(-1)).sum(1)                # (B, D)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                              label_smoothing=self.ls, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class PairwiseContrastLoss(nn.Module):
    """Extra penalty specifically for a chosen confused pair of classes
    (Fear<->Happy by default — the largest single confusion in the
    confusion matrix). Boosts the loss when the model's top-2 competing
    logits are exactly this pair, sharpening that specific decision
    boundary without disturbing other classes.

    NOTE: adding a second instance of this for a different pair (tried:
    Happy<->Neutral, stacked via .boost()) was attempted and made things
    worse in practice — see README experiment log before adding more."""
    def __init__(self, class_a: int, class_b: int, extra_weight: float = 1.5):
        super().__init__()
        self.a, self.b = class_a, class_b
        self.w = extra_weight

    def boost(self, logits, targets, loss_per_sample):
        """Per-sample boosted loss, NOT reduced to a mean, so multiple
        instances could in principle be chained before one final .mean()."""
        with torch.no_grad():
            is_pair_class = (targets == self.a) | (targets == self.b)
            top2 = logits.topk(2, dim=1).indices
            confused = ((top2[:, 0] == self.a) | (top2[:, 0] == self.b)) & \
                       ((top2[:, 1] == self.a) | (top2[:, 1] == self.b))
            boost_mask = (is_pair_class & confused).float()
        return loss_per_sample * (1.0 + self.w * boost_mask)

    def forward(self, logits, targets, base_loss_per_sample):
        return self.boost(logits, targets, base_loss_per_sample).mean()
