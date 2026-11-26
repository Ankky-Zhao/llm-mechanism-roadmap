import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

m = causal_mask(4)
assert m.dtype == torch.bool and m.tolist() == [[True, False, False, False], [True, True, False, False],
                                                [True, True, True, False], [True, True, True, True]]

torch.manual_seed(0)
s = torch.randn(4, 4, dtype=torch.float64)
w = masked_softmax(s, m)
assert torch.allclose(w.sum(-1), torch.ones(4, dtype=torch.float64)), "rows must still sum to 1"
assert (w[~m] == 0).all(), "future positions must get exactly zero weight"
assert w[0, 0] == 1.0, "the first position can only attend to itself"
assert torch.allclose(w[3], torch.softmax(s[3], -1)), "the last row sees everything, so it is an ordinary softmax"

# A tiny two-layer attention-only model, with and without the mask.
T, d = 8, 16
W = [torch.randn(3, d, d, dtype=torch.float64) / math.sqrt(d) for _ in range(2)]
def tiny(x, masked):
    for wq, wk, wv in W:
        scores = (x @ wq) @ (x @ wk).T / math.sqrt(d)
        mask = causal_mask(T) if masked else torch.ones(T, T, dtype=torch.bool)
        x = x + masked_softmax(scores, mask) @ (x @ wv)
    return x
x = torch.randn(T, d, dtype=torch.float64)
assert future_leak(lambda t: tiny(t, True), x, pos=5) < 1e-12, "with a causal mask, earlier outputs must not change at all"
assert future_leak(lambda t: tiny(t, False), x, pos=5) > 1e-3, "without the mask, earlier outputs do depend on later inputs"
assert torch.equal(x, x.clone()) and future_leak(lambda t: t, x, pos=5) == 0.0

# The real thing: GPT-2 is causal through all 12 layers.
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
ids = tok("The capital of France is Paris and the capital of Germany is Berlin", return_tensors="pt").input_ids[0]
leak = future_leak(lambda t: gpt2(t[None]).logits[0], ids, pos=6)
assert leak < 1e-3, f"GPT-2 logits before position 6 changed by {leak}"
print(f"m23_mask: all tests passed (GPT-2 leak {leak:.1e})")
