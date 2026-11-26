import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()

text = "The capital of France is"
ids = tok(text, return_tensors="pt").input_ids
assert ids.tolist() == [[464, 3139, 286, 4881, 318]], "five tokens: The / capital / of / France / is"

with torch.no_grad():
    ref = gpt2(ids).logits
out = manual_logits(gpt2, ids)
assert out.shape == (1, 5, 50257), f"one row of 50257 scores per input token, got {tuple(out.shape)}"
assert torch.allclose(out, ref, atol=1e-3), "your step-by-step logits differ from model(input_ids).logits"

top = next_token_top_k(gpt2, tok, text, k=5)
assert len(top) == 5 and all(isinstance(s, str) and isinstance(p, float) for s, p in top)
probs = [p for _, p in top]
assert probs == sorted(probs, reverse=True), "most likely first"
ref_p = torch.softmax(ref[0, -1], -1)
assert top[0][0] == tok.decode(ref_p.argmax()) and abs(top[0][1] - ref_p.max().item()) < 1e-4, "use the logits of the last position"
assert 0 < sum(probs) < 1, "five tokens out of 50257 cannot hold all the probability"
print("m01_forward: all tests passed. Top 5 after", repr(text))
for s, p in top:
    print(f"  {s!r:12} {p:.3f}")
