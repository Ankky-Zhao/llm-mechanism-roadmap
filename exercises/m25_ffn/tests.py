import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
torch.manual_seed(0)
x = torch.randn(2, 5, 768)
for layer in (0, 11):
    mlp = gpt2.transformer.h[layer].mlp
    with torch.no_grad():
        ref = mlp(x)
    assert torch.allclose(gpt2_mlp(x, mlp), ref, atol=1e-3), f"layer {layer}: differs from GPT-2's MLP"

d, h = 16, 40
xs = torch.randn(3, d, dtype=torch.float64)
wg, wu = torch.randn(d, h, dtype=torch.float64), torch.randn(d, h, dtype=torch.float64)
wd = torch.randn(h, d, dtype=torch.float64)
_silu = F.silu
F.silu = lambda *a, **k: (_ for _ in ()).throw(AssertionError("write SiLU yourself"))
try:
    out = swiglu_ffn(xs, wg, wu, wd)
finally:
    F.silu = _silu
assert out.shape == (3, d)
assert torch.allclose(out, (F.silu(xs @ wg) * (xs @ wu)) @ wd, atol=1e-10), "SwiGLU formula is wrong"

assert ffn_param_count(768, 3072, "mlp") == 2 * 768 * 3072
assert ffn_param_count(768, 2048, "swiglu") == 3 * 768 * 2048
assert swiglu_hidden_for_same_params(768) == 2048, "8d/3: this is where the factor 2/3 in LLaMA-style configs comes from"
assert ffn_param_count(768, swiglu_hidden_for_same_params(768), "swiglu") == ffn_param_count(768, 4 * 768, "mlp")
print("m25_ffn: all tests passed")
