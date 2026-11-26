import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
w = torch.randn(200, 300)

# 1. Integers in range, and the largest weight is reproduced exactly.
for bits in (8, 4):
    q, s = quantize_absmax(w, bits)
    qmax = 2 ** (bits - 1) - 1
    assert torch.equal(q, q.round()) and q.abs().max() == qmax and q.unique().numel() <= 2 * qmax + 1
    back = dequantize(q, s)
    assert (back - w).abs().max() <= s / 2 + 1e-6, "rounding moves a weight by at most half a step"
    assert torch.isclose(back.abs().max(), w.abs().max())

# 2. The rounding error behaves like uniform noise with variance scale^2 / 12.
q, s = quantize_absmax(w, 8)
mse = ((dequantize(q, s) - w) ** 2).mean().item()
assert abs(mse / (s.item() ** 2 / 12) - 1) < 0.05, "mean squared error should be close to scale^2 / 12"
q4, s4 = quantize_absmax(w, 4)
ratio = ((dequantize(q4, s4) - w) ** 2).mean().item() / mse
assert 200 < ratio < 450, f"four bits fewer make the step about 18 times larger, the error about 330 times: {ratio:.0f}"

# 3. Per-channel scales protect the other channels from one outlier.
v = torch.randn(200, 300) * 0.02
v[5, 7] = 3.0                                                  # a single very large weight in column 7
qt, st = quantize_absmax(v, 4)
qc, sc = quantize_per_channel(v, 4, channel_dim=1)
assert sc.shape == (1, 300) and torch.equal(qc, qc.round()) and qc.abs().max() <= 7
err_t = ((dequantize(qt, st) - v) ** 2).mean().item()
err_c = ((dequantize(qc, sc) - v) ** 2).mean().item()
assert (qt != 0).sum() == 1, "with one scale for the whole tensor everything except the outlier rounds to zero"
assert err_c < err_t / 5
assert quantize_per_channel(v, 4, channel_dim=0)[1].shape == (200, 1)

# 4. GPT-2 at different bit widths
tok = AutoTokenizer.from_pretrained("gpt2")
TEXT = ("The city council met on Tuesday evening to discuss the new budget. Several residents spoke about the condition of local roads, "
        "and others asked for more funding for the public library. After two hours of debate, the council voted to approve the plan with minor changes. "
        "The mayor said that construction would begin in the spring and that the work was expected to take about six months. "
        "Local businesses welcomed the decision, although some owners worried that the roadworks would keep customers away during the summer.")
ids = tok(TEXT, return_tensors="pt").input_ids

def perplexity(bits=None, per_channel=False):
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    n = quantize_gpt2_(model, bits, per_channel) if bits else 0
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0, :-1], -1)
    nll = -logp.gather(-1, ids[0, 1:, None]).mean().item()
    return math.exp(nll), n

base, _ = perplexity()
p8, n = perplexity(8)
assert n == 84_934_656, n                                      # the 48 weight matrices of the 12 blocks
p8c, _ = perplexity(8, True)
p4, _ = perplexity(4)
p4c, _ = perplexity(4, True)
assert p8 < base * 1.25 and p8c < base * 1.05, "8 bits should cost little"
assert p4 > base * 20, "4 bits with one scale per tensor should break the model"
assert p4c < base * 3 and p4c < p4 / 10, "per-channel scales should rescue most of it"

print(f"m61_quant: all tests passed ({n:,} weights quantized)")
print(f"  float32                      perplexity {base:8.2f}")
print(f"  8 bits, one scale per tensor  perplexity {p8:8.2f}")
print(f"  8 bits, one scale per channel perplexity {p8c:8.2f}")
print(f"  4 bits, one scale per tensor  perplexity {p4:8.2f}")
print(f"  4 bits, one scale per channel perplexity {p4c:8.2f}")
