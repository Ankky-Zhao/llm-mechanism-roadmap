import math
import torch

# MultiHeadAttention (module 2.2) and MiniGPT (module 2.5) are already defined above this cell.
# MiniGPT.forward recomputes every position at every step. Here the model is run one token at a time,
# and the keys and values of earlier tokens are kept in a cache.
#
# The cache of one layer is a tuple (k, v), each of shape (B, n_head, t, d_head), or None when empty.


def attention_step(attn, x_new, cache):
    """Attention for ONE new position, reusing the keys and values of earlier positions.

    attn:   a MultiHeadAttention module (use attn.qkv, attn.proj, attn.n_head, attn.d_head)
    x_new:  (B, 1, d), the input of the new position only
    cache:  (k, v) of the earlier positions, or None
    returns (out (B, 1, d), new_cache)
    """
    # TODO 1: q, k, v of the new position: attn.qkv(x_new).chunk(3, dim=-1), then split into heads -> (B, h, 1, d_head)
    # TODO 2: if there is a cache, append the new k and v to it along the position dimension (dim=2)
    # TODO 3: scores = q @ k^T / sqrt(d_head)  -> (B, h, 1, t+1). No causal mask is needed. Why?
    # TODO 4: softmax, weighted sum of v, glue the heads together, attn.proj
    raise NotImplementedError


@torch.no_grad()
def forward_step(model, token, pos, caches):
    """One step of MiniGPT on a single token.

    token:  (B, 1) token ids of the new position
    pos:    its position index (int)
    caches: a list with one cache per block
    returns (logits (B, V) for the next token, new list of caches)
    """
    # TODO 5: x = token embedding + position embedding of `pos`
    # TODO 6: for every block (Pre-Norm, as in MiniGPT):
    #             a, cache = attention_step(block.attn, block.ln1(x), cache);  x = x + a
    #             x = x + block.mlp(block.ln2(x))
    # TODO 7: final LayerNorm and head
    raise NotImplementedError


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens):
    """Greedy generation with the cache. idx: (B, T). Returns (B, T + max_new_tokens).

    Feed the prompt one token at a time to fill the cache, then generate. Every token passes
    through the model exactly once.
    """
    # TODO 8
    raise NotImplementedError
