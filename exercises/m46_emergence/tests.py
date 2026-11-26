import math

assert abs(exact_match(0.9, 1) - 0.9) < 1e-12 and abs(exact_match(0.9, 10) - 0.9 ** 10) < 1e-12
assert biggest_jump([0.1, 0.2, 0.25, 0.7, 0.8]) == 0.7 - 0.25
assert threshold_scale([1, 10, 100], [0.2, 0.6, 0.9]) == 10 and threshold_scale([1, 10], [0.1, 0.2]) is None

sizes = [10 ** (6 + 0.5 * i) for i in range(13)]                # 1e6 ... 1e12 parameters
p = [per_token_accuracy(n) for n in sizes]
assert biggest_jump(p) < 0.06, "per-token accuracy improves in small, even steps"

em1 = [exact_match(q, 1) for q in p]
em30 = [exact_match(q, 30) for q in p]
assert em1 == p
# The same models, scored on a 30-token answer with exact match:
assert max(em30[:8]) < 0.02, "small models look as if they had no ability at all"
assert em30[-1] > 0.9, "the largest model nearly always succeeds"
assert biggest_jump(em30) > 5 * biggest_jump(p), "the curve has a sudden jump although nothing sudden happened underneath"
assert threshold_scale(sizes, em30) > 100 * threshold_scale(sizes, em1), "the ability seems to 'appear' at a much larger scale"

# A continuous metric on the same task shows the smooth trend again: expected number of correct tokens.
expected_correct = [30 * q for q in p]
assert biggest_jump(expected_correct) / 30 < 0.06
print("m46_emergence: all tests passed")
print("  log10(params):", [round(math.log10(n), 1) for n in sizes])
print("  per-token acc:", [round(q, 2) for q in p])
print("  exact match, 30 tokens:", [round(q, 3) for q in em30])
