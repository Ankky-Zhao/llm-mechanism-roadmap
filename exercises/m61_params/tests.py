# 1. SmolLM2-135M, a small Llama-style model. The config values are copied from its config.json.
SMOL = {"vocab_size": 49152, "hidden_size": 576, "intermediate_size": 1536, "num_hidden_layers": 30,
        "num_attention_heads": 9, "num_key_value_heads": 3, "tie_word_embeddings": True}
n = llama_param_count(SMOL)
assert n == 134_515_008, f"the Hub reports 134,515,008 parameters for SmolLM2-135M, got {n:,}"

# 2. Untying the output matrix adds exactly one more V x d table.
assert llama_param_count({**SMOL, "tie_word_embeddings": False}) - n == 49152 * 576
# 3. Without grouped-query attention (as many key/value heads as query heads) k and v are full d x d matrices.
full = llama_param_count({**SMOL, "num_key_value_heads": 9})
assert full - n == 30 * 2 * 576 * (576 - 192)

# 4. Memory
assert model_bytes(1_000_000, 16) == 2_000_000 and model_bytes(1_000_000, 4) == 500_000
assert abs(model_bytes(n, 32) / 1e6 - 538.1) < 0.1                     # about 538 MB in float32

# 5. What fits where. Parameter counts as reported on the Hub on 2026-11-25.
QWEN_27B = 27_781_427_952          # Qwen/Qwen3.8-27B
assert precisions_that_fit(QWEN_27B, 24) == [4]                         # a 24 GB consumer GPU
assert precisions_that_fit(QWEN_27B, 48) == [8, 4]
assert precisions_that_fit(QWEN_27B, 80) == [16, 8, 4]
assert precisions_that_fit(QWEN_27B, 12) == []
assert precisions_that_fit(n, 1) == [16, 8, 4]

print(f"m61_params: all tests passed (SmolLM2-135M: {n:,} parameters)")
for bits in (32, 16, 8, 4):
    print(f"  Qwen3.8-27B at {bits:2d} bits per weight: {model_bytes(QWEN_27B, bits) / 1e9:6.1f} GB")
