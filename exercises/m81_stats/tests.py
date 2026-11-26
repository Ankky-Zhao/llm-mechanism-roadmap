from sklearn.metrics import cohen_kappa_score

# 1. Kappa, by hand and against scikit-learn
a = ["y", "y", "y", "y", "n", "n", "n", "n", "y", "n"]
b = ["y", "y", "y", "n", "n", "n", "n", "y", "y", "n"]
assert abs(cohen_kappa(a, b) - 0.6) < 1e-9                      # p_o = 0.8, p_e = 0.5
assert cohen_kappa(a, a) == 1.0
skew_a = ["ok"] * 18 + ["bad", "ok"]
skew_b = ["ok"] * 18 + ["ok", "bad"]
k = cohen_kappa(skew_a, skew_b)
assert sum(x == y for x, y in zip(skew_a, skew_b)) / 20 == 0.9 and k < 0.01, "90% raw agreement, and no agreement beyond chance"
three_a, three_b = list("aabbccabca"), list("abbbccacca")
for x, y in ((a, b), (skew_a, skew_b), (three_a, three_b)):
    assert abs(cohen_kappa(x, y) - cohen_kappa_score(x, y)) < 1e-9

# 2. Bootstrap confidence intervals
few = [1] * 16 + [0] * 4                                        # 16 of 20 correct
lo, hi = bootstrap_ci(few)
assert 0.55 <= lo <= 0.68 and 0.92 <= hi <= 1.0, (lo, hi)
many = [1] * 800 + [0] * 200                                    # the same 80%, on 1000 items
lo2, hi2 = bootstrap_ci(many)
assert 0.76 <= lo2 <= 0.79 and 0.81 <= hi2 <= 0.84
assert (hi - lo) > 5 * (hi2 - lo2), "fifty times the data, about seven times narrower"
assert bootstrap_ci(few, seed=1) == bootstrap_ci(few, seed=1), "the same seed must give the same interval"
lo8, hi8 = bootstrap_ci([1, 1, 1, 1, 0, 0, 0, 0])              # 4 of 8: the sample size of several experiments in this roadmap
assert hi8 - lo8 >= 0.6

# 3. Comparing two systems on the same items
A = [1] * 16 + [0] * 4
B = [1] * 14 + [0] * 6
p_small = paired_bootstrap(A, B)
assert 0.6 < p_small < 0.99, "16 against 14 of 20 is suggestive, not settled"
A_big, B_big = A * 25, B * 25                                   # the same rates on 500 items
assert paired_bootstrap(A_big, B_big) > 0.999
assert abs(paired_bootstrap(A, A) - 0.0) < 1e-9, "a system is never strictly better than itself"

print("m81_stats: all tests passed")
print(f"  kappa with 90% raw agreement on a skewed label distribution: {k:.2f}")
print(f"  95% interval for 16 of 20 correct:   [{lo:.2f}, {hi:.2f}]")
print(f"  95% interval for 800 of 1000 correct: [{lo2:.2f}, {hi2:.2f}]")
print(f"  95% interval for 4 of 8 correct:     [{lo8:.2f}, {hi8:.2f}]")
print(f"  16 vs 14 of 20 on the same items: A ahead in {p_small:.0%} of resamples")
