import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
z = torch.randn(4, 7, dtype=torch.float64) * 5
cases = {
    "random": z,
    "huge": torch.tensor([[1000.0, 1000.0, 1000.0], [1e4, 0.0, -1e4]], dtype=torch.float64),
    "very negative": torch.tensor([[-1e4, -1e4], [-800.0, -805.0]], dtype=torch.float64),
}
# Compute the references first, then switch the built-ins off.
refs = {k: (torch.logsumexp(v, -1), torch.log_softmax(v, -1), torch.softmax(v, -1)) for k, v in cases.items()}
ref_dim0 = torch.log_softmax(z, 0)

def _banned(*a, **k):
    raise AssertionError("built-in softmax / logsumexp is not allowed in this exercise")

_saved = [(m, n, getattr(m, n)) for m in (torch, F, torch.Tensor) for n in ("logsumexp", "softmax", "log_softmax") if hasattr(m, n)]
for m, n, _ in _saved:
    setattr(m, n, _banned)
try:
    for name, v in cases.items():
        lse, ls, sm = refs[name]
        assert torch.allclose(logsumexp(v), lse, atol=1e-10), f"logsumexp wrong on '{name}' (nan or inf means overflow)"
        assert torch.allclose(log_softmax(v), ls, atol=1e-10), f"log_softmax wrong on '{name}'"
        assert torch.allclose(softmax(v), sm, atol=1e-10), f"softmax wrong on '{name}'"
    assert logsumexp(z, keepdim=True).shape == (4, 1)
    assert torch.allclose(log_softmax(z, dim=0), ref_dim0, atol=1e-10), "the dim argument is ignored"
    p = softmax(z)
    assert torch.allclose(p.sum(-1), torch.ones(4, dtype=torch.float64))
    # Adding a constant to every logit changes nothing.
    assert torch.allclose(softmax(z + 123.0), p, atol=1e-10)
    # Two equal logits share the mass, even at -10000.
    assert abs(log_softmax(cases["very negative"])[0, 0].item() + math.log(2)) < 1e-10
finally:
    for m, n, f in _saved:
        setattr(m, n, f)
print("m13_stable_softmax: all tests passed")
