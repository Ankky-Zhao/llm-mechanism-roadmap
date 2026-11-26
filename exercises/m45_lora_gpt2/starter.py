import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# This exercise reuses build_example, collate, answer and perplexity from module 4.2.
# GPT-2 stores its attention projection in a "Conv1D" layer:  y = x @ weight + bias,
# with weight of shape (in_features, out_features), the transpose of nn.Linear.


class LoRAConv1D(nn.Module):
    """LoRA around one of GPT-2's Conv1D layers:  y = base(x) + (alpha / r) * x @ A @ B"""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        d_in, d_out = base.weight.shape
        self.base, self.scale = base, alpha / r
        # TODO 1: A (d_in, r) small random (randn * 0.01),  B (r, d_out) zeros, both nn.Parameter
        raise NotImplementedError

    def forward(self, x):
        # TODO 2
        raise NotImplementedError


def add_lora(model, r=8, alpha=16):
    """Freeze every parameter of GPT-2, then wrap the c_attn layer of all 12 blocks. Returns the model."""
    # TODO 3: for p in model.parameters(): p.requires_grad_(False)
    #         for block in model.transformer.h: block.attn.c_attn = LoRAConv1D(block.attn.c_attn, r, alpha)
    raise NotImplementedError


def count_parameters(model):
    """Returns (trainable, total)."""
    # TODO 4
    raise NotImplementedError


def lora_train(model, tok, pairs, steps=12, lr=2e-3):
    """Like sft_train from module 4.2, but the optimizer only receives the trainable parameters.

    Returns the list of losses.
    """
    # TODO 5
    raise NotImplementedError
