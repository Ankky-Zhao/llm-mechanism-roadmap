import torch
from transformers import AutoModelForCausalLM

gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
torch.manual_seed(0)
x = torch.randn(2, 9, 768)
for layer in (0, 5, 11):
    attn = gpt2.transformer.h[layer].attn
    with torch.no_grad():
        ref = attn(x)
        ref = ref[0] if isinstance(ref, tuple) else ref
    out = gpt2_attention(x, attn)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-3), f"layer {layer}: max difference {(out - ref).abs().max():.4f}"
print("m22_gpt2attn: all tests passed. You have re-implemented GPT-2's attention layer.")
