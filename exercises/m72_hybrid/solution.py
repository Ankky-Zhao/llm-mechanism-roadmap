from collections import defaultdict


def ranking(scores):
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def rrf(rankings, k=60):
    fused = defaultdict(float)
    for r in rankings:
        for position, doc in enumerate(r, start=1):
            fused[doc] += 1.0 / (k + position)
    return sorted(fused, key=lambda d: -fused[d])


def recall_at_k(rankings, gold, k):
    return sum(g in r[:k] for r, g in zip(rankings, gold)) / len(gold)


def mrr(rankings, gold):
    return sum(1.0 / (r.index(g) + 1) if g in r else 0.0 for r, g in zip(rankings, gold)) / len(gold)
