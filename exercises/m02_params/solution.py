import torch
from transformers import AutoModelForCausalLM


def count_by_component(model):
    counts = {"token_embedding": 0, "position_embedding": 0, "attention": 0, "mlp": 0, "layernorm": 0}
    for name, p in model.transformer.named_parameters():
        if name.startswith("wte"):
            key = "token_embedding"
        elif name.startswith("wpe"):
            key = "position_embedding"
        elif ".attn." in name:
            key = "attention"
        elif ".mlp." in name:
            key = "mlp"
        else:                                   # h.*.ln_1, h.*.ln_2, ln_f
            key = "layernorm"
        counts[key] += p.numel()
    counts["total"] = sum(counts.values())
    return counts


def formula_count(V, n_ctx, d, L):
    embeddings = V * d + n_ctx * d
    attention = (3 * d * d + 3 * d) + (d * d + d)          # Q, K, V in one matrix, then the output projection
    mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)            # up to 4d, back down to d
    layernorms = 2 * (2 * d)                               # a scale and a shift, twice per block
    return embeddings + L * (attention + mlp + layernorms) + 2 * d
