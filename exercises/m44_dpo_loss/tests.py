import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, V = 3, 6, 11
logits = torch.randn(B, T, V, dtype=torch.float64)
tokens = torch.randint(0, V, (B, T))
mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0]], dtype=torch.bool)
lp = sequence_logprob(logits, tokens, mask)
assert lp.shape == (B,) and (lp < 0).all()
ref = torch.stack([sum(torch.log_softmax(logits[b, t], -1)[tokens[b, t]] for t in range(T) if mask[b, t]) for b in range(B)])
assert torch.allclose(lp, ref, atol=1e-10), "sum the log-probabilities of the unmasked tokens"

pw, pl = torch.tensor([-5.0, -7.0], dtype=torch.float64), torch.tensor([-6.0, -4.0], dtype=torch.float64)
rw, rl = torch.tensor([-5.5, -6.0], dtype=torch.float64), torch.tensor([-5.0, -5.0], dtype=torch.float64)
u = 0.5 * ((pw - rw) - (pl - rl))
assert torch.allclose(dpo_loss(pw, pl, rw, rl, beta=0.5), -torch.log(torch.sigmoid(u)).mean(), atol=1e-12)

# 1. before any training the policy equals the reference, and the loss is exactly ln 2
assert abs(dpo_loss(rw, rl, rw, rl).item() - math.log(2)) < 1e-12
# 2. raising the winner or lowering the loser reduces the loss
base = dpo_loss(pw, pl, rw, rl, beta=0.5)
assert dpo_loss(pw + 1, pl, rw, rl, beta=0.5) < base and dpo_loss(pw, pl - 1, rw, rl, beta=0.5) < base
# 3. only the difference to the reference matters: a constant shift of both responses changes nothing
assert torch.allclose(dpo_loss(pw + 3, pl + 3, rw, rl, beta=0.5), base, atol=1e-12)
# 4. the gradient pushes the two log-probabilities apart, harder when the model ranks the pair wrongly
a = torch.tensor([-5.0, -5.0], dtype=torch.float64, requires_grad=True)       # log pi(y_w)
b = torch.tensor([-9.0, -1.0], dtype=torch.float64, requires_grad=True)       # log pi(y_l): pair 0 ranked right, pair 1 wrong
z = torch.zeros(2, dtype=torch.float64)
dpo_loss(a, b, z, z, beta=1.0).backward()
assert (a.grad < 0).all() and (b.grad > 0).all(), "descent raises log pi(y_w) and lowers log pi(y_l)"
assert a.grad[1].abs() > 10 * a.grad[0].abs(), "the wrongly ranked pair gets the larger gradient"

assert torch.allclose(implicit_reward(pw, rw, 0.5) - implicit_reward(pl, rl, 0.5), u)
print("m44_dpo_loss: all tests passed")
