import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))       # zero, so the update starts at 0

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def delta_weight(self):
        return self.scale * (self.B @ self.A)

    def merge(self):
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def lora_param_count(d_in, d_out, r):
    return r * (d_in + d_out)
