import torch


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dimension. x: (..., d), weight and bias: (d,).

    Subtract the mean, divide by the standard deviation (biased variance, eps inside
    the square root), then scale by weight and shift by bias.
    Do not call F.layer_norm or nn.LayerNorm.
    """
    # TODO 1
    raise NotImplementedError


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm over the last dimension: divide by sqrt(mean(x^2) + eps), then scale by weight.

    No mean subtraction and no bias.
    """
    # TODO 2
    raise NotImplementedError
