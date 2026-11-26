from collections import defaultdict

# BM25, tokenize, embed and dense_scores from exercise 1 are defined above.


def ranking(scores):
    """Document indices sorted by score, best first."""
    # TODO 1
    raise NotImplementedError


def rrf(rankings, k=60):
    """Reciprocal rank fusion of several rankings (each a list of document indices, best first).

    fused_score(d) = sum over the rankings of 1 / (k + position of d), positions counted from 1.
    A document missing from a ranking gets nothing from it. Returns the fused ranking, best first.
    Only positions are used, so scores on different scales (BM25 and cosine) can be combined.
    """
    # TODO 2
    raise NotImplementedError


def recall_at_k(rankings, gold, k):
    """Fraction of queries whose gold document is among the first k results.
    rankings: one ranking per query. gold: the index of the right document for each query."""
    # TODO 3
    raise NotImplementedError


def mrr(rankings, gold):
    """Mean reciprocal rank: the average of 1 / (rank of the gold document), 0 if it is missing."""
    # TODO 4
    raise NotImplementedError
