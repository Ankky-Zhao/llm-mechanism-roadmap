import math
import torch
from transformers import AutoModelForCausalLM

# One attention layer of GPT-2 small is  attn = model.transformer.h[i].attn  with
#   attn.c_attn  the combined Q, K, V projection.  weight (768, 2304), bias (2304,)
#   attn.c_proj  the output projection.            weight (768, 768),  bias (768,)
# These are "Conv1D" layers, which compute  y = x @ weight + bias
# (note: no transpose, unlike nn.Linear).  GPT-2 small has 12 heads of size 64.


@torch.no_grad()
def gpt2_attention(x, attn, n_head=12):
    """Recompute attn(x) from its weights. x: (B, T, 768). Returns (B, T, 768).

    GPT-2 is causal: position i may only attend to positions <= i.
    """
    B, T, d = x.shape
    # TODO 1: qkv = x @ attn.c_attn.weight + attn.c_attn.bias, split into q, k, v
    # TODO 2: split into heads, compute causal scaled dot-product attention
    # TODO 3: merge the heads, apply c_proj
    raise NotImplementedError
