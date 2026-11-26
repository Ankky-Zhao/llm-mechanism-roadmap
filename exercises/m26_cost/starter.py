import math
import time
import torch


def layer_multiplications(T, d):
    """Scalar multiplications in ONE Transformer layer for a sequence of length T (biases, norms ignored).

    Returns a dict with integer entries:
      "projections": the four d x d matrices of attention (Q, K, V, output), applied to T positions
      "scores":      Q K^T over all heads together
      "mixing":      weights times values, over all heads together
      "mlp":         d -> 4d -> d, applied to T positions
      "linear":      the part that grows like T      (projections + mlp)
      "quadratic":   the part that grows like T^2    (scores + mixing)
    """
    # TODO 1
    raise NotImplementedError


def crossover_length(d):
    """The sequence length at which the quadratic part equals the linear part."""
    # TODO 2
    raise NotImplementedError


def attention_matrix_bytes(T, n_head, n_layer, bytes_per_number=2, batch=1):
    """Memory needed to hold the attention weights of every layer at once (float16 by default)."""
    # TODO 3
    raise NotImplementedError


def fit_exponent(xs, ys):
    """Least-squares slope of log(y) against log(x). If y = c * x^p this returns p."""
    # TODO 4: with u = log x and v = log y,  slope = sum((u - mean u)(v - mean v)) / sum((u - mean u)^2)
    raise NotImplementedError


def time_attention(T, d=64, repeats=3):
    """Seconds for one single-head attention over T positions (best of `repeats`). Nothing to change."""
    q, k, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        torch.softmax(q @ k.T / math.sqrt(d), dim=-1) @ v
        best = min(best, time.perf_counter() - t0)
    return best
