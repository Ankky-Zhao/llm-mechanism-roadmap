import math, random

c, p = fit_power_law([1e6, 1e7, 1e8, 1e9], [5.0 * x ** -0.076 for x in (1e6, 1e7, 1e8, 1e9)])
assert abs(c - 5.0) < 1e-6 and abs(p + 0.076) < 1e-9
rng = random.Random(0)
xs = [10 ** (6 + 0.25 * i) for i in range(17)]
ys = [3.0 * x ** -0.1 * math.exp(rng.gauss(0, 0.02)) for x in xs]
c, p = fit_power_law(xs, ys)
assert abs(p + 0.1) < 0.01, f"with 2% noise the exponent should still be recovered, got {p:.3f}"

assert abs(chinchilla_loss(1e9, 1e11) - (1.69 + 406.4 / 1e9 ** 0.34 + 410.7 / 1e11 ** 0.28)) < 1e-12
assert chinchilla_loss(1e12, 1e15) > 1.69, "the loss never goes below the irreducible term E"
assert chinchilla_loss(2e9, 1e11) < chinchilla_loss(1e9, 1e11) and chinchilla_loss(1e9, 2e11) < chinchilla_loss(1e9, 1e11)
assert flops(1e9, 1e11) == 6e20

for C in (1e19, 1e21, 1e23):
    N, D = optimal_allocation(C)
    assert abs(flops(N, D) / C - 1) < 1e-9, "the allocation must use exactly the budget"
    best = chinchilla_loss(N, D)
    for k in range(-40, 41):                       # scan other ways of splitting the same budget
        N2 = N * 10 ** (k / 20); D2 = C / (6 * N2)
        assert chinchilla_loss(N2, D2) >= best - 1e-12, "another split of the same budget does better"

# how the optimum moves with the budget
(N1, D1), (N2, D2) = optimal_allocation(1e20), optimal_allocation(1e22)
a = math.log(N2 / N1) / math.log(100); b = math.log(D2 / D1) / math.log(100)
assert abs(a - 0.28 / 0.62) < 1e-9 and abs(b - 0.34 / 0.62) < 1e-9
assert abs(a - 0.45) < 0.01 and abs(b - 0.55) < 0.01, "parameters and tokens grow at nearly the same rate"
print("m46_scaling: all tests passed")
for C in (1e19, 1e21, 1e23):
    N, D = optimal_allocation(C)
    print(f"  C = {C:.0e} FLOPs: N = {N:.2e} parameters, D = {D:.2e} tokens, loss {chinchilla_loss(N, D):.3f}")
