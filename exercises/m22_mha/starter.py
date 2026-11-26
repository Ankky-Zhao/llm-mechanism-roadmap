import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention.

    self.qkv  maps d_model -> 3 * d_model  (queries, keys, values stacked in that order)
    self.proj maps d_model -> d_model      (the output projection W_O)
    """

    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """x: (B, T, d_model).  mask: bool (T, T), True = may attend.
        Returns (output (B, T, d_model), weights (B, n_head, T, T))."""
        B, T, d = x.shape
        # TODO 1: q, k, v = self.qkv(x) split into three chunks of size d      -> each (B, T, d)
        # TODO 2: reshape each to (B, T, n_head, d_head), then swap to (B, n_head, T, d_head)
        # TODO 3: scores (B, n_head, T, T), scaled by sqrt(d_head); apply the mask; softmax
        # TODO 4: weighted sum of values (B, n_head, T, d_head)
        # TODO 5: swap back to (B, T, n_head, d_head), merge to (B, T, d), apply self.proj
        raise NotImplementedError


def mha_param_count(d_model, n_head):
    """Number of parameters of the module above (with biases)."""
    # TODO 6: does n_head appear in the answer?
    raise NotImplementedError
