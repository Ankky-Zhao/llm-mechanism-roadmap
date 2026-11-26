import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, Tq, Tk, dk, dv = 2, 5, 7, 16, 12
q = torch.randn(B, Tq, dk, dtype=torch.float64)
k = torch.randn(B, Tk, dk, dtype=torch.float64)
v = torch.randn(B, Tk, dv, dtype=torch.float64)

out, w = scaled_dot_product_attention(q, k, v)
assert out.shape == (B, Tq, dv) and w.shape == (B, Tq, Tk), "check the shapes of output and weights"
assert torch.allclose(w.sum(-1), torch.ones(B, Tq, dtype=torch.float64)), "each query's weights must sum to 1"
assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-10), "differs from the PyTorch implementation"

# with a mask: True = may attend
mask = torch.rand(B, Tq, Tk) > 0.4
mask[..., 0] = True                                   # every query can see at least one key
out_m, w_m = scaled_dot_product_attention(q, k, v, mask)
assert torch.allclose(out_m, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-10), "masked result differs from PyTorch"
assert (w_m[~mask] == 0).all(), "masked keys must get exactly zero weight"

# the scaling really is sqrt(d_k): identical q and k rows give a score of |q|^2 / sqrt(d_k)
one = torch.ones(1, 4, dtype=torch.float64)
_, w1 = scaled_dot_product_attention(one, torch.cat([one, 0 * one]), torch.zeros(2, 1, dtype=torch.float64))
assert torch.allclose(w1, torch.softmax(torch.tensor([[2.0, 0.0]], dtype=torch.float64), -1)), "divide by sqrt(d_k), not d_k"

# self-attention on a single sequence
x = torch.randn(6, 8, dtype=torch.float64)
wq, wk, wv = (torch.randn(8, 4, dtype=torch.float64) for _ in range(3))
o, a = self_attention(x, wq, wk, wv)
assert o.shape == (6, 4) and a.shape == (6, 6)
assert torch.allclose(o, F.scaled_dot_product_attention(x @ wq, x @ wk, x @ wv), atol=1e-10)
print("m21_attention: all tests passed")
