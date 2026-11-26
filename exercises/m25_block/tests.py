import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()

torch.manual_seed(0)
x = torch.randn(1, 6, 768)
for layer in (0, 7):
    block = gpt2.transformer.h[layer]
    with torch.no_grad():
        ref = block(x)
        ref = ref[0] if isinstance(ref, tuple) else ref
    assert torch.allclose(gpt2_block(x, block), ref, atol=1e-3), f"block {layer} differs. Is the LayerNorm applied before each sublayer?"

ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    ref = gpt2(ids).logits
out = gpt2_logits(gpt2, ids)
assert out.shape == ref.shape
assert torch.allclose(out, ref, atol=5e-3), f"logits differ by up to {(out - ref).abs().max():.4f}"
assert out[0, -1].argmax() == ref[0, -1].argmax()
print("m25_block: all tests passed. Every step of GPT-2's forward pass is now your own code.")
print("  next token:", repr(tok.decode(out[0, -1].argmax())))
