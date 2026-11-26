import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 0. Floating-point addition is not associative. This is the root of everything below.
assert (0.1 + 0.2) + 0.3 != 0.1 + (0.2 + 0.3)

assert abs(top2_gap(torch.tensor([1.0, 3.5, 3.0, -2.0])) - 0.5) < 1e-6
assert first_divergence([1, 2, 3], [1, 2, 3]) is None and first_divergence([1, 2, 3], [1, 5, 3]) == 1

tok = AutoTokenizer.from_pretrained("gpt2")
m32 = AutoModelForCausalLM.from_pretrained("gpt2").eval()
m16 = AutoModelForCausalLM.from_pretrained("gpt2").to(torch.float16).eval()      # same weights, half precision
N = 30
prompts = ["Scientists have discovered that", "The weather today is", "In machine learning, attention is", "The capital of France is"]

# 1. Same model, same precision, same machine: greedy decoding repeats exactly.
ids = tok(prompts[0], return_tensors="pt").input_ids
t_a, g_a = greedy_trace(m32, ids, N)
t_b, _ = greedy_trace(m32, ids, N)
assert len(t_a) == N and len(g_a) == N and all(g >= 0 for g in g_a)
assert first_divergence(t_a, t_b) is None, "nothing random happens in greedy decoding"
hf = m32.generate(ids, do_sample=False, max_new_tokens=N, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
assert t_a == hf

# 2. The same weights in half precision give slightly different logits.
with torch.no_grad():
    l32, l16 = m32(ids).logits[0, -1], m16(ids).logits[0, -1].float()
err = (l32 - l16).abs().max().item()
assert 1e-4 < err < 2.0, err

# 3. Whether that changes the text depends on how close the decisions are.
rows, diverged = [], 0
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids
    t32, g32 = greedy_trace(m32, ids, N)
    t16, _ = greedy_trace(m16, ids, N)
    d = first_divergence(t32, t16)
    diverged += d is not None
    rows.append((p, d, g32[d] if d is not None else min(g32), tok.decode(t32), tok.decode(t16)))
assert diverged >= 1, "at least one of these prompts should diverge between float32 and float16"
print(f"m51_temp0: all tests passed (largest logit difference between float32 and float16 on one prompt: {err:.3f})")
for p, d, gap, a, b in rows:
    if d is None:
        print(f"  {p!r}: identical for {N} tokens, smallest top-2 gap {gap:.3f}")
    else:
        print(f"  {p!r}: diverges at step {d}, where the top-2 gap was {gap:.4f}")
        print(f"      float32: {a!r}")
        print(f"      float16: {b!r}")
