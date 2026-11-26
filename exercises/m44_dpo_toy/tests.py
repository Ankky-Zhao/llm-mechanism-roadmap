import torch

torch.manual_seed(0)
K, beta = 6, 0.5
logp_ref = torch.log_softmax(torch.randn(K, dtype=torch.float64), -1)
r = torch.randn(K, dtype=torch.float64)                              # the hidden reward the annotators follow

g = torch.Generator().manual_seed(0)
w, l = sample_preferences(r, 20000, g)
pi = dpo_train(logp_ref, w, l, beta)
assert pi.shape == (K,) and abs(pi.sum().item() - 1) < 1e-9

# The closed-form optimum of RLHF from module 4.3: pi* proportional to pi_ref * exp(r / beta)
star = torch.softmax(logp_ref + r / beta, -1)
err = (pi - star).abs().max().item()
assert err < 0.03, f"DPO should land on the RLHF optimum without ever seeing the reward, max error {err:.3f}"

# The implicit reward beta * log(pi / pi_ref) recovers the hidden reward up to a constant.
implied = beta * (pi.log() - logp_ref)
diff = (implied - implied.mean()) - (r - r.mean())
assert diff.abs().max() < 0.15, "the implicit reward should match the hidden one up to a shift"

# With no data the policy stays where it started.
none = torch.tensor([], dtype=torch.long)
assert torch.allclose(dpo_train(logp_ref, w[:1], l[:1], beta, steps=0), logp_ref.exp())
print(f"m44_dpo_toy: all tests passed (max error to the RLHF optimum {err:.4f})")
print("  reference :", [round(v, 3) for v in logp_ref.exp().tolist()])
print("  DPO result:", [round(v, 3) for v in pi.tolist()])
print("  RLHF optimum:", [round(v, 3) for v in star.tolist()])
