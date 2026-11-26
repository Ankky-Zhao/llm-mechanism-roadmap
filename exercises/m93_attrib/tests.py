import time
import torch

model, tok = load_gpt2()
clean, corrupt, ans, wrong = ioi_batch(tok)
clean_logits, cache = run_with_cache(model, clean)

t0 = time.time()
acts, grads = run_with_grad(model, corrupt, ans, wrong)
t_attr = time.time() - t0

# 1. Activations equal those of an ordinary run, gradients have the same shapes.
corrupt_logits, corrupt_cache = run_with_cache(model, corrupt)
assert set(acts) == set(corrupt_cache) == set(grads)
for k in acts:
    assert torch.allclose(acts[k], corrupt_cache[k], rtol=1e-4, atol=1e-3), k
    assert grads[k].shape == acts[k].shape and not acts[k].requires_grad
assert sum(len(b.attn.c_proj._forward_pre_hooks) for b in model.transformer.h) == 0

# 2. The gradient is a real derivative: a small step along it changes the metric as predicted.
c, x = logit_diff(clean_logits, ans, wrong), logit_diff(corrupt_logits, ans, wrong)
g = grads[("resid", 8)]
eps = 1e-3 / g.norm()
def hook(module, args, out):
    y = block_output(out) + eps * g
    return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
h = model.transformer.h[8].register_forward_hook(hook)
with torch.no_grad():
    moved = logit_diff(model(corrupt).logits, ans, wrong).mean().item()
h.remove()
predicted = (eps * g.pow(2).sum()).item()
assert abs((moved - x.mean().item()) - predicted) < 0.05 * abs(predicted) + 1e-5, (moved - x.mean().item(), predicted)

# 3. Heads: the estimate against real patching.
scale = (c.mean() - x.mean()).item()
A = attribution_heads(model, cache, acts, grads, scale)
assert A.shape == (12, 12)
t0 = time.time()
M = head_patching_map(model, clean, corrupt, ans, wrong)
t_patch = time.time() - t0
r = torch.corrcoef(torch.stack([A.flatten(), M.flatten()]))[0, 1].item()
assert r > 0.9, r
top_true = set(M.flatten().abs().argsort(descending=True)[:8].tolist())
top_est = set(A.flatten().abs().argsort(descending=True)[:8].tolist())
assert len(top_true & top_est) >= 6

# 4. Residual stream: the estimate against real patching.
AR_ = attribution_residual(model, cache, acts, grads, scale)
MR = residual_patching_map(model, clean, corrupt, ans, wrong)
assert AR_.shape == MR.shape == (12, 14)
assert AR_[:, :S2].abs().max() < 1e-4, "no difference, no attribution"

print(f"heads: one forward and one backward pass {t_attr:.1f} s, 144 patching runs {t_patch:.1f} s")
print(f"correlation between estimate and real effect over the 144 heads: {r:.3f}, {len(top_true & top_est)} of the 8 strongest heads agree")
worst = (A - M).abs().flatten().argsort(descending=True)[:4].tolist()
print("largest errors: " + ", ".join(f"{i // 12}.{i % 12} estimate {A.flatten()[i]:+.2f} real {M.flatten()[i]:+.2f}" for i in worst))
print("residual stream at S2 and END, estimate / real")
for l in range(12):
    print(f"  layer {l:>2}   S2 {AR_[l, S2].item():>+6.2f} / {MR[l, S2].item():>+5.2f}    END {AR_[l, END].item():>+6.2f} / {MR[l, END].item():>+5.2f}")
