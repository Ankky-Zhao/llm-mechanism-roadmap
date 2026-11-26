import math
import torch


def attention_step(attn, x_new, cache):
    B, _, d = x_new.shape
    q, k, v = attn.qkv(x_new).chunk(3, dim=-1)                                    # each (B, 1, d)
    split = lambda t: t.view(B, 1, attn.n_head, attn.d_head).transpose(1, 2)      # (B, h, 1, d_head)
    q, k, v = split(q), split(k), split(v)
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)                                       # (B, h, t+1, d_head)
        v = torch.cat([cache[1], v], dim=2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(attn.d_head)                     # (B, h, 1, t+1), no mask needed
    out = torch.softmax(scores, dim=-1) @ v                                       # (B, h, 1, d_head)
    out = out.transpose(1, 2).contiguous().view(B, 1, d)
    return attn.proj(out), (k, v)


@torch.no_grad()
def forward_step(model, token, pos, caches):
    x = model.tok_emb(token) + model.pos_emb(torch.tensor([pos]))                 # (B, 1, d)
    new_caches = []
    for block, cache in zip(model.blocks, caches):
        a, cache = attention_step(block.attn, block.ln1(x), cache)
        x = x + a
        x = x + block.mlp(block.ln2(x))
        new_caches.append(cache)
    return model.head(model.ln_f(x))[:, 0], new_caches


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens):
    caches = [None] * len(model.blocks)
    for pos in range(idx.shape[1]):                                               # prefill, one token at a time
        logits, caches = forward_step(model, idx[:, pos:pos + 1], pos, caches)
    for i in range(max_new_tokens):
        nxt = logits.argmax(-1, keepdim=True)                                     # (B, 1)
        idx = torch.cat([idx, nxt], dim=1)
        if i < max_new_tokens - 1:                                                # the last token needs no further pass
            logits, caches = forward_step(model, nxt, idx.shape[1] - 1, caches)
    return idx
