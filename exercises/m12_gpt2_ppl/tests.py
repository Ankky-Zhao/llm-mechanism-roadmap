import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()

text = "The capital of France is Paris, and the capital of Germany is Berlin."
ids = tok(text, return_tensors="pt").input_ids

nll = token_nlls(gpt2, ids)
assert nll.shape == (ids.shape[1] - 1,), f"expected {ids.shape[1] - 1} scores, got {tuple(nll.shape)}"
assert (nll > 0).all(), "a negative log-likelihood cannot be negative"

# Hugging Face computes the same mean when labels are passed.
with torch.no_grad():
    ref = gpt2(ids, labels=ids).loss
assert torch.allclose(nll.mean(), ref, atol=1e-4), f"mean NLL {nll.mean():.4f} vs reference {ref:.4f}"

ppl = text_perplexity(gpt2, tok, text)
assert abs(ppl - ref.exp().item()) < 1e-2

# Shuffling the words destroys the structure the model relies on.
words = text.split()
random.Random(0).shuffle(words)
ppl_shuffled = text_perplexity(gpt2, tok, " ".join(words))
assert ppl_shuffled > 3 * ppl, "shuffled text should be far more surprising"
print(f"m12_gpt2_ppl: all tests passed (PPL {ppl:.1f}, shuffled {ppl_shuffled:.1f})")
