import math
import torch


def sinusoidal_pe(T, d, base=10000.0):
    """The fixed positional encoding of the original Transformer.

    Returns a float64 tensor (T, d), d even, with
      PE[pos, 2i]   = sin(pos / base^(2i/d))
      PE[pos, 2i+1] = cos(pos / base^(2i/d))
    """
    # TODO 1: positions (T, 1), frequencies (d/2,), then fill even and odd columns
    raise NotImplementedError


def is_permutation_equivariant(f, x, perm):
    """True if shuffling the rows of x just shuffles the rows of f(x) the same way.

    f maps (T, d) -> (T, d'); perm is a permutation of range(T) as a long tensor.
    """
    # TODO 2: compare f(x[perm]) with f(x)[perm] using torch.allclose(..., atol=1e-8)
    raise NotImplementedError
