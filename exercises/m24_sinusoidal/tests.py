import math
import torch

T, d = 50, 16
pe = sinusoidal_pe(T, d)
assert pe.shape == (T, d) and pe.dtype == torch.float64
assert torch.allclose(pe[0], torch.tensor([0.0, 1.0] * (d // 2), dtype=torch.float64)), "position 0 is sin(0), cos(0), ..."
assert abs(pe[3, 0].item() - math.sin(3.0)) < 1e-12 and abs(pe[3, 1].item() - math.cos(3.0)) < 1e-12
assert abs(pe[7, 4].item() - math.sin(7 / 10000 ** (4 / d))) < 1e-12, "column 2i uses frequency 1 / base^(2i/d)"
assert torch.allclose(pe.pow(2).sum(-1), torch.full((T,), d / 2, dtype=torch.float64)), "every row has squared norm d/2"

# The dot product of two encodings depends only on the distance between the positions.
for k in (1, 5, 12):
    dots = (pe[:-k] * pe[k:]).sum(-1)
    assert torch.allclose(dots, dots[0].expand_as(dots)), f"PE[p] . PE[p+{k}] should be the same for every p"

# Self-attention without positional information cannot tell word order.
torch.manual_seed(0)
x = torch.randn(6, d, dtype=torch.float64)
wq, wk, wv = (torch.randn(d, d, dtype=torch.float64) / 4 for _ in range(3))
def attn(h):
    a = torch.softmax((h @ wq) @ (h @ wk).T / math.sqrt(d), -1)
    return a @ (h @ wv)
perm = torch.tensor([3, 0, 5, 1, 4, 2])
assert is_permutation_equivariant(attn, x, perm), "plain self-attention is permutation equivariant"
assert not is_permutation_equivariant(lambda h: attn(h + sinusoidal_pe(6, d)), x, perm), "adding positions must break the symmetry"
assert not is_permutation_equivariant(lambda h: h.cumsum(0), x, perm)
print("m24_sinusoidal: all tests passed")
