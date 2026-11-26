import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()

r = retrieval_correct(gpt2, tok, ["Alice", "Bob", "Carol"], [17, 42, 88], 2)
assert isinstance(r, bool)
assert r is True, "GPT-2 retrieves the most recent fact"

acc = accuracy_by_position(gpt2, tok, n_items=5, trials=12, seed=0)
assert len(acc) == 5 and all(0.0 <= a <= 1.0 for a in acc)
assert acc == accuracy_by_position(gpt2, tok, n_items=5, trials=12, seed=0), "same seed, same result"
assert acc[-1] > 0.7, f"the last fact should be retrieved most of the time, got {acc[-1]:.2f}"
assert acc[-1] > acc[0] + 0.4, f"the same kind of fact is retrieved far less often at the start ({acc[0]:.2f}) than at the end ({acc[-1]:.2f})"
print("m32_position: all tests passed")
print("  accuracy by position (first ... last):", [round(a, 2) for a in acc], " chance = 0.20")
