import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank update.

    output = base(x) + (alpha / r) * x @ A^T @ B^T
    A: (r, in_features), small random values.   B: (out_features, r), zeros.
    """

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        # TODO 1: freeze the parameters of base (requires_grad_(False))
        # TODO 2: create self.A as nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        #         and    self.B as nn.Parameter(torch.zeros(base.out_features, r))
        raise NotImplementedError

    def forward(self, x):
        # TODO 3
        raise NotImplementedError

    def delta_weight(self):
        """The update as a full matrix of the same shape as base.weight."""
        # TODO 4
        raise NotImplementedError

    def merge(self):
        """Return a plain nn.Linear whose weight is base.weight + delta_weight (same bias)."""
        # TODO 5
        raise NotImplementedError


def lora_param_count(d_in, d_out, r):
    """Trainable parameters of one LoRA update."""
    # TODO 6
    raise NotImplementedError
