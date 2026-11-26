import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)                                   # each (B, T, d)
        split = lambda t: t.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)                # (B, h, T, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v                                                        # (B, h, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, d)                     # glue the heads back together
        return self.proj(out), weights


def mha_param_count(d_model, n_head):
    return (3 * d_model * d_model + 3 * d_model) + (d_model * d_model + d_model)  # independent of n_head
