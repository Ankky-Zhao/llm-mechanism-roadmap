from transformers import AutoModelForCausalLM

gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
c = count_by_component(gpt2)
expected = {"token_embedding": 38_597_376, "position_embedding": 786_432, "attention": 28_348_416,
            "mlp": 56_669_184, "layernorm": 38_400, "total": 124_439_808}
for k, v in expected.items():
    assert c[k] == v, f"{k}: got {c[k]:,}, expected {v:,}"
assert c["total"] == sum(p.numel() for p in gpt2.parameters()), "the unembedding shares the token embedding, so it adds nothing"
assert formula_count(50257, 1024, 768, 12) == 124_439_808, "the formula should give GPT-2 small"
assert formula_count(50257, 1024, 1024, 24) == 354_823_168, "and GPT-2 medium"
print("m02_params: all tests passed")
for k, v in c.items():
    print(f"  {k:20} {v:>12,}  {100 * v / c['total']:5.1f}%")
