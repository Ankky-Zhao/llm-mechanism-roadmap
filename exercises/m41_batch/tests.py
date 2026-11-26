import math
import torch
import torch.nn.functional as F

data = torch.arange(1000)
g = torch.Generator().manual_seed(0)
x, y = make_batch(data, block_size=16, batch_size=8, generator=g)
assert x.shape == (8, 16) and y.shape == (8, 16)
assert torch.equal(y, x + 1), "y is x shifted by one position"
assert torch.equal(x[:, 1:], y[:, :-1]), "the target at position t is the input at position t + 1"
assert len(set(x[:, 0].tolist())) > 1, "start positions should be random"

torch.manual_seed(0)
B, T, V = 4, 9, 50
logits = torch.randn(B, T, V, dtype=torch.float64)
targets = torch.randint(0, V, (B, T))
ref = F.cross_entropy(logits.view(-1, V), targets.view(-1))
assert torch.allclose(lm_loss(logits, targets), ref, atol=1e-10), "differs from F.cross_entropy"

masked = targets.clone()
masked[:, :4] = -100                                    # pretend the first four positions carry no target
ref_m = F.cross_entropy(logits.view(-1, V), masked.view(-1), ignore_index=-100)
assert torch.allclose(lm_loss(logits, masked), ref_m, atol=1e-10), "ignored positions must not count, neither in the sum nor in the average"

assert abs(expected_initial_loss(50257) - 10.8249) < 1e-3
uniform = torch.zeros(B, T, V, dtype=torch.float64)
assert abs(lm_loss(uniform, targets).item() - expected_initial_loss(V)) < 1e-10

assert steps_per_epoch(1_000_000, 1024, 32) == 30
assert steps_per_epoch(10_000_000_000, 1024, 512) == 19073
print("m41_batch: all tests passed")
