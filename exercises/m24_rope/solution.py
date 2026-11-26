import math
import torch


def rope_rotate(x, positions, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)        # (d/2,)
    angles = positions.to(x.dtype)[:, None] * theta                    # (T, d/2)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]                                # the two halves of each pair
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)   # (..., T, d/2, 2)
    return out.flatten(-2)                                             # back to (..., T, d)


def rope_scores(q, k, positions):
    return rope_rotate(q, positions) @ rope_rotate(k, positions).transpose(-2, -1)
