import math
import torch


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)        # (..., T_q, T_k): one score per (query, key) pair
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))    # exp(-inf) = 0, so these keys get zero weight
    weights = torch.softmax(scores, dim=-1)                  # each query's weights sum to 1
    return weights @ v, weights                              # weighted average of the values


def self_attention(x, w_q, w_k, w_v):
    return scaled_dot_product_attention(x @ w_q, x @ w_k, x @ w_v)
