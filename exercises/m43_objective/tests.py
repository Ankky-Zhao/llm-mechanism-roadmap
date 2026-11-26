import torch

torch.manual_seed(0)
K, beta = 6, 0.5
logp_ref = torch.log_softmax(torch.randn(K, dtype=torch.float64), -1)
r = torch.randn(K, dtype=torch.float64)

# 1. the objective
assert abs(kl_regularized_objective(logp_ref, logp_ref, r, beta).item() - (logp_ref.exp() * r).sum().item()) < 1e-12, "KL of a distribution with itself is 0"
star = optimal_policy(logp_ref, r, beta)
assert abs(star.sum().item() - 1) < 1e-12
J_star = kl_regularized_objective(star.log(), logp_ref, r, beta).item()
Z = (logp_ref.exp() * torch.exp(r / beta)).sum()
assert abs(J_star - beta * torch.log(Z).item()) < 1e-10, "the optimal value is beta * log Z"
for _ in range(200):                                      # no other policy does better
    q = torch.log_softmax(torch.randn(K, dtype=torch.float64) * 2, -1)
    assert kl_regularized_objective(q, logp_ref, r, beta).item() <= J_star + 1e-10
# 2. beta controls how far the policy may move
assert torch.allclose(optimal_policy(logp_ref, r, 1e6), logp_ref.exp(), atol=1e-4), "huge beta: stay at the reference model"
assert optimal_policy(logp_ref, r, 1e-3).argmax() == r.argmax() and optimal_policy(logp_ref, r, 1e-3).max() > 0.999, "tiny beta: put everything on the highest reward"

# 3. PPO clipping
lo, ln = torch.log(torch.tensor([0.5, 0.5, 0.5, 0.5])), torch.log(torch.tensor([0.9, 0.9, 0.1, 0.1]))
A = torch.tensor([1.0, -1.0, 1.0, -1.0])
# ratios are 1.8, 1.8, 0.2, 0.2 -> per sample: min(1.8, 1.2)=1.2, min(-1.8, -1.2)=-1.8, min(0.2, 0.8)=0.2, min(-0.2, -0.8)=-0.8
assert abs(ppo_clip_objective(ln, lo, A, eps=0.2).item() - (1.2 - 1.8 + 0.2 - 0.8) / 4) < 1e-6
assert abs(ppo_clip_objective(lo, lo, A).item() - A.mean().item()) < 1e-6, "at ratio 1 the objective is the mean advantage"
x = ln.clone().requires_grad_(True)
ppo_clip_objective(x, lo, A).backward()
assert x.grad[0] == 0, "a good action whose probability already rose past 1 + eps gets no further push"
assert x.grad[1] != 0, "a bad action whose probability rose is still pushed back"

# 4. policy gradient finds the closed-form optimum
theta = logp_ref.clone()
g = torch.Generator().manual_seed(0)
for _ in range(400):
    theta = reinforce_step(theta, logp_ref, r, beta, generator=g)
err = (torch.softmax(theta, -1) - star).abs().max().item()
assert err < 0.03, f"after 400 steps the policy should be close to the closed form, max error {err:.3f}"
print(f"m43_objective: all tests passed (max error to the closed form {err:.4f})")
print("  reference:", [round(v, 3) for v in logp_ref.exp().tolist()])
print("  rewards  :", [round(v, 2) for v in r.tolist()])
print("  optimum  :", [round(v, 3) for v in star.tolist()])
