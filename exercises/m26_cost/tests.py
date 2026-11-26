import math

c = layer_multiplications(1024, 768)
assert c["projections"] == 4 * 1024 * 768 ** 2 and c["mlp"] == 8 * 1024 * 768 ** 2
assert c["scores"] == c["mixing"] == 1024 ** 2 * 768
assert c["linear"] == 12 * 1024 * 768 ** 2 and c["quadratic"] == 2 * 1024 ** 2 * 768
assert c["linear"] > c["quadratic"], "at GPT-2 small's full context the linear part still dominates"

assert crossover_length(768) == 4608
x = layer_multiplications(crossover_length(768), 768)
assert x["linear"] == x["quadratic"]
big = layer_multiplications(100_000, 4096)
assert big["quadratic"] > 4 * big["linear"], "at 100k tokens the quadratic part dominates"

# doubling T doubles the linear part and quadruples the quadratic part
a, b = layer_multiplications(2000, 512), layer_multiplications(4000, 512)
assert b["linear"] == 2 * a["linear"] and b["quadratic"] == 4 * a["quadratic"]

assert attention_matrix_bytes(1024, 12, 12) == 12 * 12 * 1024 ** 2 * 2                 # GPT-2 small, about 0.3 GB
assert attention_matrix_bytes(8192, 32, 32) / 2 ** 30 == 128.0, "8k context, 32 heads, 32 layers: 128 GiB if stored naively"

assert abs(fit_exponent([1, 2, 4, 8], [3, 12, 48, 192]) - 2.0) < 1e-9
assert abs(fit_exponent([10, 100, 1000], [5, 50, 500]) - 1.0) < 1e-9

Ts = [512, 1024, 2048, 4096]
secs = [time_attention(T) for T in Ts]
p = fit_exponent(Ts, secs)
assert p > 1.5, f"measured exponent {p:.2f}: attention time should grow clearly faster than linearly"
print(f"m26_cost: all tests passed (measured time exponent {p:.2f})")
