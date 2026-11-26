import random


def cohen_kappa(a, b):
    n = len(a)
    labels = set(a) | set(b)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if p_e == 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def bootstrap_ci(values, n_resamples=2000, level=0.95, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_resamples))
    lo = means[int((1 - level) / 2 * n_resamples)]
    hi = means[int((1 + level) / 2 * n_resamples) - 1]
    return lo, hi


def paired_bootstrap(scores_a, scores_b, n_resamples=2000, seed=0):
    rng = random.Random(seed)
    n = len(scores_a)
    wins = 0
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        wins += sum(scores_a[i] for i in idx) > sum(scores_b[i] for i in idx)
    return wins / n_resamples
