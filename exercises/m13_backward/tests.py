import torch
import torch.nn.functional as F

torch.manual_seed(0)
N, D, V = 8, 5, 11

# 1. Jacobian of softmax
z = torch.randn(V, dtype=torch.float64)
J_ref = torch.autograd.functional.jacobian(lambda t: torch.softmax(t, -1), z)
with torch.no_grad():
    J = softmax_jacobian(z)
assert J.shape == (V, V)
assert torch.allclose(J, J_ref, atol=1e-10), "Jacobian does not match autograd"
assert torch.allclose(J.sum(0), torch.zeros(V, dtype=torch.float64), atol=1e-10), "columns of J sum to 0"

# 2. gradient of mean cross-entropy with respect to logits
logits = torch.randn(N, V, dtype=torch.float64, requires_grad=True)
targets = torch.randint(0, V, (N,))
F.cross_entropy(logits, targets).backward()
with torch.no_grad():
    g = ce_grad_logits(logits.detach().clone(), targets)
assert torch.allclose(g, logits.grad, atol=1e-10), "expected (p - y) / N"
assert torch.allclose(g.sum(-1), torch.zeros(N, dtype=torch.float64), atol=1e-10), "each row of p - y sums to 0"

# 3. one linear layer, all three gradients
x = torch.randn(N, D, dtype=torch.float64, requires_grad=True)
W = torch.randn(V, D, dtype=torch.float64, requires_grad=True)
b = torch.randn(V, dtype=torch.float64, requires_grad=True)
ref_loss = F.cross_entropy(x @ W.T + b, targets)
ref_loss.backward()
with torch.no_grad():
    loss, dW, db, dx = linear_ce_backward(x.detach(), W.detach(), b.detach(), targets)
assert torch.allclose(loss, ref_loss, atol=1e-10)
assert torch.allclose(dW, W.grad, atol=1e-10), "dW should be dlogits^T x"
assert torch.allclose(db, b.grad, atol=1e-10), "db should be the column sums of dlogits"
assert torch.allclose(dx, x.grad, atol=1e-10), "dx should be dlogits W"
print("m13_backward: all tests passed")
