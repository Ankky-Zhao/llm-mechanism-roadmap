import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Sampling follows the probabilities: draw many times and count.
z = torch.tensor([2.0, 1.0, 0.1])
g = torch.Generator().manual_seed(0)
draws = torch.stack([sample_next(z, 1.0, g) for _ in range(20000)])
freq = torch.bincount(draws, minlength=3) / 20000
assert torch.allclose(freq, torch.softmax(z, -1), atol=0.015), f"frequencies {freq.tolist()} should be close to softmax(z) = (0.659, 0.242, 0.099)"

# 2. Temperature reshapes the distribution.
g = torch.Generator().manual_seed(0)
cold = torch.stack([sample_next(z, 0.05, g) for _ in range(500)])
assert (cold == 0).all(), "at a very low temperature only the top token is drawn"
g = torch.Generator().manual_seed(0)
hot = torch.stack([sample_next(z, 100.0, g) for _ in range(20000)])
assert torch.allclose(torch.bincount(hot, minlength=3) / 20000, torch.full((3,), 1 / 3), atol=0.015), "at a very high temperature all tokens are about equally likely"

# 3. On the real model.
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
ids = tok("The meaning of life is", return_tensors="pt").input_ids

a = sample_generate(gpt2, ids, 40, temperature=1.0, seed=1)
b = sample_generate(gpt2, ids, 40, temperature=1.0, seed=1)
c = sample_generate(gpt2, ids, 40, temperature=1.0, seed=2)
assert a.shape == (1, ids.shape[1] + 40)
assert torch.equal(a, b), "same seed, same output"
assert not torch.equal(a, c), "a different seed should give a different continuation"

greedy = gpt2.generate(ids, do_sample=False, max_new_tokens=40, pad_token_id=tok.eos_token_id)
assert torch.equal(sample_generate(gpt2, ids, 40, temperature=1e-4, seed=3), greedy), "temperature near 0 should reproduce greedy decoding"

def _repeats(x, n=4):
    seen, r = set(), 0
    for i in range(len(x) - n + 1):
        gram = tuple(x[i:i + n]); r += gram in seen; seen.add(gram)
    return r
assert _repeats(a[0].tolist()) < _repeats(greedy[0].tolist()), "sampled text should repeat less than greedy text"
print("m01_sample: all tests passed")
print("  greedy :", repr(tok.decode(greedy[0])))
print("  seed 1 :", repr(tok.decode(a[0])))
print("  seed 2 :", repr(tok.decode(c[0])))
