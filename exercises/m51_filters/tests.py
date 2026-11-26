import math
import torch
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

torch.manual_seed(0)
logits = torch.randn(50) * 2

# 1. Temperature rescales, it does not reorder.
assert torch.allclose(apply_temperature(logits, 0.5), logits * 2)
assert apply_temperature(logits, 3.0).argmax() == logits.argmax()
ent = lambda l: -(torch.softmax(l, -1) * torch.log_softmax(l, -1)).sum().item()
assert ent(apply_temperature(logits, 0.5)) < ent(logits) < ent(apply_temperature(logits, 2.0)), "lower temperature, lower entropy"

# 2. Top-k keeps exactly k tokens and agrees with the Hugging Face implementation.
for k in (1, 5, 20):
    mine = top_k_filter(logits, k)
    assert torch.isfinite(mine).sum() == k
    assert torch.equal(mine, TopKLogitsWarper(top_k=k)(None, logits[None])[0])

# 3. Top-p on a case that can be checked by hand.
hand = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
assert torch.isfinite(top_p_filter(hand, 0.7)).tolist() == [True, True, False, False]      # 0.5 < 0.7, 0.5+0.3 >= 0.7
assert torch.isfinite(top_p_filter(hand, 0.85)).tolist() == [True, True, True, False]
assert torch.isfinite(top_p_filter(hand, 0.1)).tolist() == [True, False, False, False]     # the top token always stays
shuffled = hand[[2, 0, 3, 1]]
assert torch.isfinite(top_p_filter(shuffled, 0.7)).tolist() == [False, True, False, True], "restore the original order"

# 4. Top-p agrees with the Hugging Face implementation on random logits.
for p in (0.3, 0.6, 0.9, 0.95):
    mine = top_p_filter(logits, p)
    assert torch.equal(mine, TopPLogitsWarper(top_p=p)(None, logits[None])[0]), p
    kept = torch.softmax(logits, -1)[torch.isfinite(mine)]
    assert kept.sum() >= p and kept.sum() - kept.min() < p, "the kept set is the smallest one reaching p"
assert torch.allclose(apply_temperature(logits, 0.7), TemperatureLogitsWarper(0.7)(None, logits[None])[0])

# 5. The number of tokens kept by top-p adapts to the distribution; top-k does not.
peaked, flat = torch.cat([torch.tensor([8.0]), torch.linspace(0, -0.5, 49)]), torch.zeros(50)
assert torch.isfinite(top_p_filter(peaked, 0.9)).sum() == 1 and torch.isfinite(top_p_filter(flat, 0.9)).sum() == 45
assert torch.isfinite(top_k_filter(peaked, 10)).sum() == 10

# 6. sample_token
assert sample_token(logits, temperature=0) == logits.argmax()
g = torch.Generator().manual_seed(0)
draws = torch.stack([sample_token(logits, 1.0, top_k=5, generator=g) for _ in range(300)])
assert set(draws.tolist()) <= set(logits.topk(5).indices.tolist()), "top-k = 5 can only ever produce 5 different tokens"
g = torch.Generator().manual_seed(0)
cold = torch.stack([sample_token(logits, 0.1, generator=g) for _ in range(200)])
assert (cold == logits.argmax()).float().mean() > 0.95, "a low temperature is close to greedy"
print("m51_filters: all tests passed")
