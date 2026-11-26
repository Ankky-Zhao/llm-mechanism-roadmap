import math
import torch


def rope_rotate(x, positions, base=10000.0):
    """Rotary position embedding (RoPE).

    x: (..., T, d) with d even.  positions: (T,) tensor of positions (they need not start at 0).
    Split the last dimension into d/2 pairs (x[2i], x[2i+1]). Rotate pair i of the token
    at position m by the angle  m * theta_i,  theta_i = base^(-2i/d):
        out[2i]   = x[2i] * cos - x[2i+1] * sin
        out[2i+1] = x[2i] * sin + x[2i+1] * cos
    Returns a tensor of the same shape as x.
    """
    # TODO 1: theta (d/2,), angles (T, d/2) = positions[:, None] * theta
    # TODO 2: take the even and odd components of x and apply the rotation
    # TODO 3: interleave the two results back into shape (..., T, d)
    raise NotImplementedError


def rope_scores(q, k, positions):
    """Attention scores (before softmax and scaling) with RoPE: rotate q and k, then q k^T."""
    # TODO 4
    raise NotImplementedError
