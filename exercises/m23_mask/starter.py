import torch


def causal_mask(T):
    """Bool tensor (T, T). Entry [i, j] is True when position i may attend to position j."""
    # TODO 1: position i sees itself and everything before it
    raise NotImplementedError


def masked_softmax(scores, mask):
    """Softmax over the last dimension, with zero weight wherever mask is False."""
    # TODO 2: fill with -inf before the softmax (not zero afterwards: the rows must still sum to 1)
    raise NotImplementedError


@torch.no_grad()
def future_leak(f, x, pos):
    """Does the output before `pos` depend on the input from `pos` onwards?

    f maps a tensor (T, ...) to a tensor (T, ...), one output row per position.
    Replace x[pos:] by random values of the same shape and dtype, run f on both
    versions, and return the largest absolute change in the outputs at positions < pos.
    (For an integer x, draw random token ids with torch.randint_like(x[pos:], 50257).)
    """
    # TODO 3
    raise NotImplementedError
