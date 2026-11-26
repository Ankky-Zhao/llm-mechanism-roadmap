import torch
from transformers import AutoModelForCausalLM


def count_by_component(model):
    """Where do GPT-2's parameters live?

    Returns a dict with integer counts for the keys
      "token_embedding", "position_embedding", "attention", "mlp", "layernorm", "total".
    Walk over model.transformer.named_parameters(); the names look like
      wte.weight, wpe.weight, h.0.ln_1.weight, h.0.attn.c_attn.weight, h.0.mlp.c_fc.bias, ln_f.weight
    """
    counts = {"token_embedding": 0, "position_embedding": 0, "attention": 0, "mlp": 0, "layernorm": 0}
    for name, p in model.transformer.named_parameters():
        # TODO 1: decide from `name` which bucket p.numel() belongs to
        raise NotImplementedError
    counts["total"] = sum(counts.values())
    return counts


def formula_count(V, n_ctx, d, L):
    """The same total from a closed-form formula.

    V vocabulary size, n_ctx maximum context length, d hidden size, L number of blocks.
    The MLP hidden size is 4d. The unembedding shares its weights with the token embedding.
    """
    # TODO 2: embeddings + L * (attention + mlp + two LayerNorms) + final LayerNorm
    raise NotImplementedError
