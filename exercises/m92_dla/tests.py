import torch

model, tok = load_gpt2()
enc = lambda s: tok(s, return_tensors="pt").input_ids[0]
prompt = "The Eiffel Tower is located in the city of"
ids = enc(prompt)
paris, london = enc(" Paris")[0].item(), enc(" London")[0].item()

names, parts = components(model, ids)

# 1. One embedding term and two terms per layer.
assert names[:3] == ["embed", "attn0", "mlp0"] and names[-1] == "mlp11" and len(names) == 25
assert parts.shape == (25, 768)
assert all(len(b.attn._forward_hooks) == 0 and len(b.mlp._forward_hooks) == 0 for b in model.transformer.h)

# 2. The residual stream is the sum of these terms.
resid = residual_stream(model, ids)
assert torch.allclose(parts.sum(0), resid[-1], atol=1e-3), "the parts add up to the final residual stream"
assert torch.allclose(parts[:5].sum(0), resid[2], atol=1e-3), "and the first 1 + 2l parts to the stream after l blocks"

# 3. The contributions and the constant add up to the logit difference of the model.
contrib, const = direct_logit_attribution(model, parts, paris, london)
with torch.no_grad():
    logits = model(ids[None]).logits[0, -1]
true = (logits[paris] - logits[london]).item()
assert contrib.shape == (25,)
assert abs(contrib.sum().item() + const - true) < 1e-2, (contrib.sum().item() + const, true)

# 4. Antisymmetry.
c2, k2 = direct_logit_attribution(model, parts, london, paris)
assert torch.allclose(c2, -contrib, atol=1e-5) and abs(k2 + const) < 1e-6

print(prompt)
print(f"logit(' Paris') - logit(' London') = {true:.2f}, sum of the parts {contrib.sum().item():.2f} plus constant {const:.2f}")
order = contrib.abs().argsort(descending=True)[:8].tolist()
print("largest direct contributions:")
for j in order:
    print(f"  {names[j]:<7} {contrib[j]:+.2f}")
att = sum(contrib[j].item() for j in range(25) if names[j].startswith("attn"))
mlp = sum(contrib[j].item() for j in range(25) if names[j].startswith("mlp"))
print(f"all attention sublayers {att:+.2f}, all MLPs {mlp:+.2f}, embedding {contrib[0]:+.2f}")

# A second prompt.
p2 = "The opposite of hot is"
cold, hot = enc(" cold")[0].item(), enc(" hot")[0].item()
n2, parts2 = components(model, enc(p2))
c3, k3 = direct_logit_attribution(model, parts2, cold, hot)
with torch.no_grad():
    lg = model(enc(p2)[None]).logits[0, -1]
assert abs(c3.sum().item() + k3 - (lg[cold] - lg[hot]).item()) < 1e-2
print(p2)
print(f"logit(' cold') - logit(' hot') = {(lg[cold] - lg[hot]).item():.2f}, largest parts " +
      ", ".join(f"{n2[j]} {c3[j]:+.2f}" for j in c3.abs().argsort(descending=True)[:6].tolist()))
