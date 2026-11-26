import torch

model, tok = load_gpt2()
clean, corrupt, ans, wrong = ioi_batch(tok)
clean_logits, cache = run_with_cache(model, clean)
with torch.no_grad():
    corrupt_logits = model(corrupt).logits

# 1. Patching a head changes exactly its slice.
out = patch_head(model, corrupt, cache, 8, 10)
assert out.shape == corrupt_logits.shape
assert not torch.allclose(out, corrupt_logits, atol=1e-3)
assert sum(len(b.attn.c_proj._forward_pre_hooks) for b in model.transformer.h) == 0

# 2. Patching the 12 heads of a layer one after the other in the same run equals replacing the whole input of c_proj.
def patch_layer(layer):
    def hook(module, args): return (cache[("z", layer)].clone(),)
    h = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad(): return model(corrupt).logits
    finally: h.remove()
hs = []
for hd in range(12):
    dd = 64
    def hook(module, args, hd=hd):
        z = args[0].clone(); z[:, :, hd * dd:(hd + 1) * dd] = cache[("z", 9)][:, :, hd * dd:(hd + 1) * dd]; return (z,)
    hs.append(model.transformer.h[9].attn.c_proj.register_forward_pre_hook(hook))
with torch.no_grad(): all12 = model(corrupt).logits
for h in hs: h.remove()
assert torch.allclose(all12, patch_layer(9), atol=1e-4)

# 3. Restricting the patch to a position.
c, x = logit_diff(clean_logits, ans, wrong), logit_diff(corrupt_logits, ans, wrong)
eff = lambda logits: normalized(logit_diff(logits, ans, wrong), c, x)
e_all, e_end, e_s2 = eff(patch_head(model, corrupt, cache, 8, 10)), eff(patch_head(model, corrupt, cache, 8, 10, END)), eff(patch_head(model, corrupt, cache, 8, 10, S2))
assert abs(e_all - e_end) < 0.02 and abs(e_s2) < 0.02, "head 8.10 acts at the last position"

# 4. The full map (144 forward passes).
M = head_patching_map(model, clean, corrupt, ans, wrong)
assert M.shape == (12, 12)
flat = M.flatten()
top = [(i // 12, i % 12) for i in flat.argsort(descending=True)[:8].tolist()]
neg = [(i // 12, i % 12) for i in flat.argsort()[:3].tolist()]
assert top[0] == (8, 10), top
assert set(neg[:2]) == {(10, 7), (11, 10)} and M[10, 7] < -0.1, "the two negative name movers work against the answer"
assert M[:3].abs().max() < 0.1

ROLE = {(2, 2): "previous token", (4, 11): "previous token", (0, 1): "duplicate token", (3, 0): "duplicate token", (0, 10): "duplicate token",
        (5, 5): "induction", (6, 9): "induction", (5, 8): "induction", (5, 9): "induction",
        (7, 3): "S-inhibition", (7, 9): "S-inhibition", (8, 6): "S-inhibition", (8, 10): "S-inhibition",
        (9, 9): "name mover", (9, 6): "name mover", (10, 0): "name mover",
        (10, 7): "negative name mover", (11, 10): "negative name mover",
        (10, 10): "backup name mover", (10, 6): "backup name mover", (10, 2): "backup name mover", (10, 1): "backup name mover",
        (11, 2): "backup name mover", (9, 7): "backup name mover", (9, 0): "backup name mover", (11, 9): "backup name mover"}
listed = sum((l, h) in ROLE for l, h in top)
assert listed >= 6, "most of the strongest heads belong to the circuit described by Wang et al."
print("head   effect   at END   at S2   role in Wang et al. (2022)")
for l, h in top + neg:
    at_end, at_s2 = eff(patch_head(model, corrupt, cache, l, h, END)), eff(patch_head(model, corrupt, cache, l, h, S2))
    print(f"{l}.{h:<4} {M[l, h].item():>+6.2f}   {at_end:>+6.2f}  {at_s2:>+6.2f}   {ROLE.get((l, h), 'not listed')}")
print(f"heads with |effect| > 0.05: {(M.abs() > 0.05).sum().item()} of 144")
print(f"sum over all heads {M.sum().item():.2f}")
name_movers = [(9, 9), (9, 6), (10, 0)]
print("name movers of Wang et al.: " + ", ".join(f"{l}.{h} {M[l, h].item():+.2f}" for l, h in name_movers))
