import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_absmax(w, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q, scale


def quantize_per_channel(w, bits, channel_dim):
    qmax = 2 ** (bits - 1) - 1
    other = 1 - channel_dim
    scale = w.abs().amax(dim=other, keepdim=True) / qmax          # one scale per channel
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q, scale


def dequantize(q, scale):
    return q * scale


@torch.no_grad()
def quantize_gpt2_(model, bits, per_channel):
    n = 0
    for name, p in model.named_parameters():
        if p.dim() == 2 and name.startswith("transformer.h."):
            if per_channel:
                q, s = quantize_per_channel(p, bits, channel_dim=1)   # GPT-2 stores weights as (in, out)
            else:
                q, s = quantize_absmax(p, bits)
            p.copy_(dequantize(q, s))
            n += p.numel()
    return n
