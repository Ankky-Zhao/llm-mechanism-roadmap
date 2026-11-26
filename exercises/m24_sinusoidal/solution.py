import math
import torch


def sinusoidal_pe(T, d, base=10000.0):
    pos = torch.arange(T, dtype=torch.float64)[:, None]                 # (T, 1)
    freq = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)    # (d/2,)  = 1 / base^(2i/d)
    pe = torch.zeros(T, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def is_permutation_equivariant(f, x, perm):
    return torch.allclose(f(x[perm]), f(x)[perm], atol=1e-8)
