import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
text = " ".join(["The quick brown fox jumps over the lazy dog."] * 90)
full = tok(text, return_tensors="pt").input_ids[:, :800]
N = 12

assert latency_summary([0.5, 0.1, 0.1, 0.2]) == {"ttft": 0.5, "tokens_per_second": 3 / 0.4}
assert kv_cache_bytes(12, 12, 64, 1024) == 2 * 12 * 12 * 64 * 1024 * 4            # about 75 MB for GPT-2 small at full context

rows = []
for T in (50, 200, 800):
    prompt = full[:, :T]
    a, ta = generate_no_cache(model, prompt, N)
    b, tb, cache = generate_with_cache(model, prompt, N)
    # 1. The cache changes the cost, not the result.
    assert a.shape == (1, T + N) and torch.equal(a, b), "both methods must produce the same tokens"
    assert len(ta) == N and len(tb) == N
    # 2. The cache holds every position that has passed through the model: the prompt and all but the last new token.
    pairs = cache_tensors(cache)
    assert len(pairs) == 12 and pairs[0][0].shape == (1, 12, T + N - 1, 64)
    measured = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in pairs)
    assert measured == kv_cache_bytes(12, 12, 64, T + N - 1), "the formula should match the real cache exactly"
    rows.append((T, latency_summary(ta), latency_summary(tb), measured))

# 3. Timing. The thresholds are loose because wall-clock time varies between machines.
no800, yes800 = rows[2][1], rows[2][2]
assert yes800["tokens_per_second"] > 3 * no800["tokens_per_second"], "with an 800-token prompt the cache should speed decoding up several times"
assert rows[2][2]["ttft"] > 2 * rows[0][2]["ttft"], "time to first token grows with the prompt length"
decode = [r[2]["tokens_per_second"] for r in rows]
assert max(decode) < 4 * min(decode), "with the cache, decoding speed depends only weakly on the prompt length"

print("m52_latency: all tests passed")
print("  prompt   time to first token        decoding speed (tokens/s)      cache size")
print("  tokens   no cache   with cache       no cache   with cache")
for T, s0, s1, size in rows:
    print(f"  {T:5d}   {s0['ttft']:7.3f}s   {s1['ttft']:7.3f}s      {s0['tokens_per_second']:8.1f}   {s1['tokens_per_second']:8.1f}        {size / 1e6:6.1f} MB")
