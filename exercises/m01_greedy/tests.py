import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()

assert repeated_ngrams([1, 2, 3, 4, 5, 6], n=4) == 0
assert repeated_ngrams([1, 2, 3, 4, 1, 2, 3, 4, 1], n=4) == 2, "(1,2,3,4) and (2,3,4,1) each appear a second time"

ids = tok("The meaning of life is", return_tensors="pt").input_ids
out = greedy_generate(gpt2, ids, max_new_tokens=40)
assert out.shape == (1, ids.shape[1] + 40)
assert torch.equal(out[:, :ids.shape[1]], ids), "the prompt must stay at the front"
ref = gpt2.generate(ids, do_sample=False, max_new_tokens=40, pad_token_id=tok.eos_token_id)
assert torch.equal(out, ref), "differs from model.generate(do_sample=False)"

# Greedy decoding is deterministic.
assert torch.equal(out, greedy_generate(gpt2, ids, max_new_tokens=40))

# ... and it gets stuck in a loop. This is the reason sampling exists.
reps = repeated_ngrams(out[0].tolist(), n=4)
assert reps >= 10, "greedy GPT-2 output is expected to repeat itself here"
print("m01_greedy: all tests passed")
print(f"  {reps} repeated 4-grams in:", repr(tok.decode(out[0])))
