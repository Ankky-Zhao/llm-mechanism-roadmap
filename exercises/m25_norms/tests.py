import torch
import torch.nn as nn

torch.manual_seed(0)
d = 32
x = torch.randn(4, 7, d, dtype=torch.float64) * 3 + 2

ln = nn.LayerNorm(d, dtype=torch.float64)
with torch.no_grad():
    ln.weight.uniform_(0.5, 1.5); ln.bias.uniform_(-1, 1)
assert torch.allclose(layer_norm(x, ln.weight, ln.bias), ln(x), atol=1e-10), "differs from nn.LayerNorm"

rn = nn.RMSNorm(d, eps=1e-6, dtype=torch.float64)
with torch.no_grad():
    rn.weight.uniform_(0.5, 1.5)
assert torch.allclose(rms_norm(x, rn.weight), rn(x), atol=1e-10), "differs from nn.RMSNorm"

ones, zeros = torch.ones(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
y = layer_norm(x, ones, zeros)
assert torch.allclose(y.mean(-1), torch.zeros(4, 7, dtype=torch.float64), atol=1e-9), "LayerNorm output has mean 0"
assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 7, dtype=torch.float64), atol=1e-4), "and variance 1"
z = rms_norm(x, ones)
assert torch.allclose(z.pow(2).mean(-1), torch.ones(4, 7, dtype=torch.float64), atol=1e-4), "RMSNorm output has mean square 1"
assert z.mean(-1).abs().max() > 0.1, "RMSNorm does not remove the mean"

# Both are invariant to rescaling the input, which is what keeps activations from blowing up.
assert torch.allclose(layer_norm(10 * x, ones, zeros), y, atol=1e-5)
assert torch.allclose(rms_norm(10 * x, ones), z, atol=1e-5)
# For an input that already has mean 0, the two coincide.
x0 = x - x.mean(-1, keepdim=True)
assert torch.allclose(layer_norm(x0, ones, zeros, eps=1e-6), rms_norm(x0, ones, eps=1e-6), atol=1e-10)
print("m25_norms: all tests passed")
