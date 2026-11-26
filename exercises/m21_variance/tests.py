import torch

g = lambda: torch.Generator().manual_seed(0)

# 1. Var(q . k) = d_k, and dividing by sqrt(d_k) brings it back to 1
for d in (16, 64, 256, 1024):
    v = dot_product_variance(d, generator=g())
    assert abs(v / d - 1) < 0.1, f"d_k={d}: variance {v:.1f} should be close to {d}"
    vs = dot_product_variance(d, scaled=True, generator=g())
    assert abs(vs - 1) < 0.1, f"d_k={d}: scaled variance {vs:.3f} should be close to 1"

# 2. without scaling, softmax collapses onto one key as d_k grows
raw = [mean_max_weight(d, generator=g()) for d in (16, 64, 256, 1024)]
assert raw == sorted(raw) and raw[-1] > 0.9, f"unscaled weights should get more peaked with d_k: {raw}"
sc = [mean_max_weight(d, scaled=True, generator=g()) for d in (16, 64, 256, 1024)]
assert max(sc) - min(sc) < 0.05 and max(sc) < 0.4, f"scaled weights should stay equally spread for every d_k: {sc}"

# 3. a saturated softmax passes little gradient
g_raw = score_gradient_norm(1024, generator=g())
g_sc = score_gradient_norm(1024, scaled=True, generator=g())
assert g_sc > 3 * g_raw, f"scaled gradient {g_sc:.3f} should be several times the unscaled {g_raw:.3f}"
print("m21_variance: all tests passed")
print("  mean max weight, unscaled:", [round(x, 3) for x in raw])
print("  mean max weight, scaled  :", [round(x, 3) for x in sc])
print(f"  gradient norm at d_k=1024: unscaled {g_raw:.3f}, scaled {g_sc:.3f}")
