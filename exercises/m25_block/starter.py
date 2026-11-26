import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# This exercise reuses gpt2_attention (module 2.2) and gpt2_mlp (previous exercise).
# A GPT-2 block is  block = model.transformer.h[i]  with
#   block.ln_1, block.attn, block.ln_2, block.mlp
# GPT-2 is a Pre-Norm model.


@torch.no_grad()
def gpt2_block(x, block):
    """One full Transformer block, from your own pieces. x: (B, T, 768)."""
    # TODO 1: x = x + attention(LayerNorm_1(x))     use F.layer_norm(x, (768,), ln.weight, ln.bias, ln.eps)
    # TODO 2: x = x + mlp(LayerNorm_2(x))
    raise NotImplementedError


@torch.no_grad()
def gpt2_logits(model, input_ids):
    """The complete GPT-2 forward pass without calling any GPT-2 layer's forward method.

    input_ids: (1, T).  Returns logits (1, T, 50257).
    Token embedding table: model.transformer.wte.weight   (50257, 768)
    Position embedding table: model.transformer.wpe.weight (1024, 768)
    The unembedding shares the token embedding table.
    """
    t = model.transformer
    # TODO 3: embeddings -> 12 x gpt2_block -> final LayerNorm (t.ln_f) -> x @ wte.weight.T
    raise NotImplementedError
