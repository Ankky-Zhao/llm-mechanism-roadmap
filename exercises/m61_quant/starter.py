import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Symmetric ("absmax") quantization stores every weight as a small integer q and one shared scale s,
# with w approximately equal to q * s. With b bits the integers run from -qmax to qmax, qmax = 2^(b-1) - 1.


def quantize_absmax(w, bits):
    """One scale for the whole tensor.

    scale = max|w| / qmax,   q = round(w / scale), clamped to [-qmax, qmax]
    Returns (q, scale): q is a float tensor holding integer values, scale a 0-dim tensor.
    """
    # TODO 1
    raise NotImplementedError


def quantize_per_channel(w, bits, channel_dim):
    """One scale per channel of a 2-D weight matrix.

    channel_dim = 1 means one scale for every column: the maximum is taken over the other dimension.
    Use w.abs().amax(dim=..., keepdim=True) so that the scales broadcast against w.
    Returns (q, scale) with scale of shape (1, n) or (n, 1).
    """
    # TODO 2
    raise NotImplementedError


def dequantize(q, scale):
    """Back to floating point."""
    # TODO 3
    raise NotImplementedError


@torch.no_grad()
def quantize_gpt2_(model, bits, per_channel):
    """Replace every weight matrix inside the Transformer blocks by its quantized-and-dequantized version, in place.

    Loop over model.named_parameters(); take those with p.dim() == 2 whose name starts with "transformer.h.".
    GPT-2 stores these matrices as (in, out), so the output channels are channel_dim = 1.
    Write the result back with p.copy_(...). Returns the number of weights that were quantized.
    The model still computes in float32; this simulates the loss of precision, not the memory saving.
    """
    # TODO 4
    raise NotImplementedError
