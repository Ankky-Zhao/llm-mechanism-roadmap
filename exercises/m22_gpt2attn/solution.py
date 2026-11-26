import math
import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def gpt2_attention(x, attn, n_head=12):
    B, T, d = x.shape
    q, k, v = (x @ attn.c_attn.weight + attn.c_attn.bias).split(d, dim=-1)
    split = lambda t: t.view(B, T, n_head, d // n_head).transpose(1, 2)        # (B, h, T, 64)
    q, k, v = split(q), split(k), split(v)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d // n_head)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    out = out.transpose(1, 2).contiguous().view(B, T, d)
    return out @ attn.c_proj.weight + attn.c_proj.bias
