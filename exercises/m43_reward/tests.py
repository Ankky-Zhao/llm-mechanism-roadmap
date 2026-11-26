import math
import torch

assert abs(preference_prob(torch.tensor(1.0), torch.tensor(1.0)).item() - 0.5) < 1e-12, "equal scores, even odds"
assert abs(preference_prob(torch.tensor(2.0), torch.tensor(0.0)).item() - 1 / (1 + math.exp(-2))) < 1e-7
a, b = torch.tensor(0.3), torch.tensor(-1.1)
assert abs(preference_prob(a, b).item() + preference_prob(b, a).item() - 1) < 1e-7
assert abs(preference_prob(a + 5, b + 5).item() - preference_prob(a, b).item()) < 1e-7, "only the difference of scores matters"

rw, rl = torch.tensor([2.0, 0.0, -1.0]), torch.tensor([0.0, 0.0, 3.0])
expected = -sum(math.log(1 / (1 + math.exp(-(x - y)))) for x, y in zip(rw.tolist(), rl.tolist())) / 3
assert abs(pairwise_loss(rw, rl).item() - expected) < 1e-6
assert torch.isfinite(pairwise_loss(torch.tensor([-1000.0]), torch.tensor([1000.0]))), "use logsigmoid, not log(sigmoid(...))"

# Simulated annotators: a hidden "true quality" decides each comparison, with Bradley-Terry noise.
g = torch.Generator().manual_seed(0)
N, d, M = 200, 5, 4000
X = torch.randn(N, d, generator=g, dtype=torch.float64)
w_true = torch.tensor([2.0, -1.0, 0.5, 0.0, 1.5], dtype=torch.float64)
quality = X @ w_true
i = torch.randint(0, N, (M,), generator=g); j = torch.randint(0, N, (M,), generator=g)
i_wins = torch.rand(M, generator=g, dtype=torch.float64) < torch.sigmoid(quality[i] - quality[j])
winners, losers = torch.where(i_wins, i, j), torch.where(i_wins, j, i)

w = train_reward_model(X, winners, losers)
assert w.shape == (d,)
learned = X @ w
# the learned reward should rank fresh pairs the way the true quality does
a, b = torch.randint(0, N, (5000,), generator=g), torch.randint(0, N, (5000,), generator=g)
agree = ((learned[a] > learned[b]) == (quality[a] > quality[b])).double().mean().item()
assert agree > 0.95, f"ranking agreement {agree:.3f}"
cos = torch.dot(w, w_true) / (w.norm() * w_true.norm())
assert cos > 0.98, f"the learned direction should match the hidden one, cosine {cos:.3f}"
print(f"m43_reward: all tests passed (ranking agreement {agree:.3f}, cosine {cos:.3f})")
print("  learned w:", [round(x, 2) for x in w.tolist()], " hidden w:", w_true.tolist())
