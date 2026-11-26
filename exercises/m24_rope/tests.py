import math
import torch

torch.manual_seed(0)
B, T, d = 2, 9, 16
q = torch.randn(B, T, d, dtype=torch.float64)
k = torch.randn(B, T, d, dtype=torch.float64)
pos = torch.arange(T)

r = rope_rotate(q, pos)
assert r.shape == q.shape
assert torch.allclose(r[:, 0], q[:, 0]), "position 0 is rotated by angle 0, so it must not change"
assert torch.allclose(r.norm(dim=-1), q.norm(dim=-1)), "a rotation keeps the length of every vector"

# An independent reference: treat each pair as a complex number and multiply by e^{i m theta}.
theta = 10000.0 ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
phase = torch.polar(torch.ones(T, d // 2, dtype=torch.float64), pos[:, None].double() * theta)
ref = torch.view_as_real(torch.view_as_complex(q.reshape(B, T, d // 2, 2).contiguous()) * phase).flatten(-2)
assert torch.allclose(r, ref, atol=1e-12), "differs from the complex-number reference"

# The key property: scores depend only on the distance between positions.
s0 = rope_scores(q, k, pos)
s1 = rope_scores(q, k, pos + 1000)
assert s0.shape == (B, T, T)
assert torch.allclose(s0, s1, atol=1e-9), "shifting every position by 1000 must leave the scores unchanged"
assert not torch.allclose(s0, q @ k.transpose(-2, -1)), "but the scores do differ from attention without positions"

# The same pair of vectors, placed at different distances, gets different scores.
qq, kk = q[0, 0], k[0, 0]
score = lambda m, n: (rope_rotate(qq[None], torch.tensor([m])) @ rope_rotate(kk[None], torch.tensor([n])).T).item()
assert abs(score(3, 5) - score(40, 42)) < 1e-9 and abs(score(3, 5) - score(3, 9)) > 1e-6
print("m24_rope: all tests passed")
