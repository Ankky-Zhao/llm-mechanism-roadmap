import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
original = {n: p.detach().clone() for n, p in model.named_parameters()}
ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    logits_before = model(ids).logits

model = add_lora(model, r=8, alpha=16)
with torch.no_grad():
    assert torch.allclose(model(ids).logits, logits_before, atol=1e-5), "with B = 0 the model must behave exactly as before"
trainable, total = count_parameters(model)
assert trainable == 12 * 8 * (768 + 2304) == 294912, f"trainable parameters: {trainable}"
assert trainable / total < 0.003, "less than 0.3% of the model is trained"

# The PEFT library builds the same thing: identical number of trainable parameters.
peft_model = get_peft_model(AutoModelForCausalLM.from_pretrained("openai-community/gpt2"),
                            LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True))
assert sum(p.numel() for p in peft_model.parameters() if p.requires_grad) == trainable

pairs = [("What is the capital of France?", "Paris"), ("What is the capital of Germany?", "Berlin"),
         ("What is the capital of Italy?", "Rome"), ("What is the capital of Spain?", "Madrid"),
         ("What is the capital of Japan?", "Tokyo"), ("What is the capital of China?", "Beijing"),
         ("What is the capital of Russia?", "Moscow"), ("What is the capital of Egypt?", "Cairo")]
general = "The weather was cold and the streets were quiet as people walked home from work in the evening."
ppl_before = perplexity(model, tok, general)
losses = lora_train(model, tok, pairs, steps=12, lr=2e-3)
assert losses[-1] < 0.1 * losses[0], f"loss {losses[0]:.2f} -> {losses[-1]:.3f}"
out = answer(model, tok, "What is the capital of Poland?")
assert out.startswith("ANSWER:"), f"got {out!r}"
ppl_after = perplexity(model, tok, general)

# The pretrained weights are untouched: every original tensor is bit-for-bit the same.
now = dict(model.named_parameters())
for n, p in original.items():
    key = n.replace("attn.c_attn.", "attn.c_attn.base.")
    assert torch.equal(now[key], p), f"{n} changed"
# So removing the adapters restores the original model exactly.
for block in model.transformer.h:
    block.attn.c_attn = block.attn.c_attn.base
with torch.no_grad():
    assert torch.allclose(model(ids).logits, logits_before, atol=1e-5)
assert abs(perplexity(model, tok, general) - ppl_before) < 1e-3
print("m45_lora_gpt2: all tests passed")
print(f"  trainable {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")
print(f"  held-out answer: {out!r}")
print(f"  perplexity on ordinary text: {ppl_before:.1f} -> {ppl_after:.1f} with the adapters, back to {ppl_before:.1f} after removing them")
