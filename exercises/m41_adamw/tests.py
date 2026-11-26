import math
import torch

torch.manual_seed(0)
w0 = torch.randn(5, 3, dtype=torch.float64)
target = torch.randn(5, 3, dtype=torch.float64)

# reference: torch.optim.AdamW on a simple quadratic loss
w_ref = w0.clone().requires_grad_(True)
opt = torch.optim.AdamW([w_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
p, m, v = w0.clone(), torch.zeros_like(w0), torch.zeros_like(w0)
for t in range(1, 21):
    loss = ((w_ref - target) ** 2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    grad = 2 * (p - target)
    before = p.clone()
    p, m, v = adamw_step(p, grad, m, v, t, lr=1e-2, weight_decay=0.1)
    assert not torch.equal(before, p)
    assert torch.allclose(p, w_ref.detach(), atol=1e-10), f"step {t}: differs from torch.optim.AdamW"

# the very first step moves every parameter by about lr, whatever the size of its gradient
p1, _, _ = adamw_step(torch.zeros(3, dtype=torch.float64), torch.tensor([1e-4, 1.0, -1e4], dtype=torch.float64),
                      torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 1, lr=0.01, weight_decay=0.0)
assert torch.allclose(p1, torch.tensor([-0.01, -0.01, 0.01], dtype=torch.float64), atol=1e-5), "Adam normalizes the step size per parameter"

lr = lambda s: lr_schedule(s, max_lr=1.0, warmup_steps=10, total_steps=110, min_lr=0.1)
assert abs(lr(0) - 0.1) < 1e-12 and abs(lr(4) - 0.5) < 1e-12 and abs(lr(9) - 1.0) < 1e-12, "linear warm-up"
assert abs(lr(10) - 1.0) < 1e-12, "the decay starts at the maximum"
assert abs(lr(60) - 0.55) < 1e-12, "halfway through the decay the rate is halfway between max and min"
assert abs(lr(110) - 0.1) < 1e-12 and lr(500) == 0.1
assert all(lr(s) >= lr(s + 1) for s in range(10, 110)), "monotone decay after warm-up"
print("m41_adamw: all tests passed")
