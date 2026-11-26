import math
import torch


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    q: (..., T_q, d_k)   k: (..., T_k, d_k)   v: (..., T_k, d_v)
    mask: optional bool tensor broadcastable to (..., T_q, T_k).
          True means "this query may look at this key".
    Returns (output, weights) with shapes (..., T_q, d_v) and (..., T_q, T_k).
    Do not call F.scaled_dot_product_attention.
    """
    # TODO 1: scores = q @ k^T, divided by sqrt(d_k)
    # TODO 2: where mask is False, set the score to -inf
    # TODO 3: softmax over the keys (the last dimension)
    # TODO 4: output = weights @ v
    raise NotImplementedError


def self_attention(x, w_q, w_k, w_v):
    """Single-head self-attention: queries, keys and values all come from x.

    x: (T, d)   w_q, w_k: (d, d_k)   w_v: (d, d_v)
    """
    # TODO 5: project x three times, then call scaled_dot_product_attention
    raise NotImplementedError
