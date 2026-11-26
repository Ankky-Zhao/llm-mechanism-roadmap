import torch

torch.manual_seed(0)
V = 50
model = MiniGPT(V, block_size=32, d_model=64, n_head=4, n_layer=3).eval()
idx = torch.randint(0, V, (2, 12))

# 1. Step by step with the cache gives the same logits as the full forward pass, at every position.
with torch.no_grad():
    full, _ = model(idx)                                             # (2, 12, V)
caches = [None] * len(model.blocks)
for t in range(idx.shape[1]):
    step_logits, caches = forward_step(model, idx[:, t:t + 1], t, caches)
    assert step_logits.shape == (2, V)
    assert torch.allclose(step_logits, full[:, t], atol=1e-5), f"position {t} differs from the full forward pass"

# 2. The cache holds one key and one value per layer, head and position.
assert len(caches) == 3
for k, v in caches:
    assert k.shape == (2, 4, 12, 16) and v.shape == (2, 4, 12, 16)

# 3. Cached greedy generation equals greedy generation that recomputes everything.
def generate_plain(model, idx, n):
    for _ in range(n):
        with torch.no_grad():
            logits, _ = model(idx)
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return idx

out = generate_cached(model, idx, 15)
assert out.shape == (2, 27) and torch.equal(out, generate_plain(model, idx, 15))

# 4. Count the work: with the cache every token is embedded exactly once.
calls = {"plain": 0, "cached": 0}
hook = None
def counter(name):
    def f(module, inp, out):
        calls[name] += inp[0].numel()
    return f
h = model.tok_emb.register_forward_hook(counter("plain")); generate_plain(model, idx[:1], 15); h.remove()
h = model.tok_emb.register_forward_hook(counter("cached")); generate_cached(model, idx[:1], 15); h.remove()
assert calls["cached"] == 12 + 15 - 1, calls           # every token once; the last generated token is never fed back
assert calls["plain"] == sum(range(12, 27)), calls     # 12 + 13 + ... + 26
print(f"m52_kvcache: all tests passed (token positions processed: {calls['plain']} without the cache, {calls['cached']} with it)")
