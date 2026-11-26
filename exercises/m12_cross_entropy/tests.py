import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
N, V = 64, 50
logits = torch.randn(N, V, dtype=torch.float64) * 3
targets = torch.randint(0, V, (N,))

ref = F.cross_entropy(logits, targets)
assert torch.allclose(cross_entropy(logits, targets), ref, atol=1e-10), "does not match F.cross_entropy"
assert torch.allclose(perplexity(logits, targets), ref.exp(), atol=1e-8), "perplexity should be exp(cross-entropy)"
assert torch.allclose(bits_per_token(logits, targets), ref / math.log(2), atol=1e-10), "bits = nats / ln 2"

# A model that knows nothing (uniform) has loss ln V and perplexity V.
uniform = torch.zeros(N, V, dtype=torch.float64)
assert abs(cross_entropy(uniform, targets).item() - math.log(V)) < 1e-10
assert abs(perplexity(uniform, targets).item() - V) < 1e-8

# A model that is certain and right has loss 0 and perplexity 1.
sure = torch.full((N, V), -50.0, dtype=torch.float64)
sure[torch.arange(N), targets] = 50.0
assert perplexity(sure, targets).item() < 1 + 1e-8

# Large logits must not overflow.
big = logits * 1e4
assert torch.isfinite(cross_entropy(big, targets)), "overflow: work in log space"

# Gradients flow through your implementation.
x = logits.clone().requires_grad_(True)
cross_entropy(x, targets).backward()
y = logits.clone().requires_grad_(True)
F.cross_entropy(y, targets).backward()
assert torch.allclose(x.grad, y.grad, atol=1e-10)
print("m12_cross_entropy: all tests passed")
