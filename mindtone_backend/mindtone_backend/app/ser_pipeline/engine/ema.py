"""Exponential moving average of model weights, used for both validation
and the final saved checkpoint (empirically more stable than raw weights)."""
import copy

import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.995, warmup_steps: int = 500):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        # Linearly ramp 0 -> self.decay over warmup_steps, then hold. Early
        # updates (shadow still close to random init) get a low decay so
        # the shadow catches up fast; later updates get full smoothing.
        d = self.decay * min(1.0, self.step_count / self.warmup_steps)
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(d).add_(m.data, alpha=1.0 - d)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)
