import math
import re
import string
from collections import Counter


def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction, reference):
    p, r = normalize(prediction).split(), normalize(reference).split()
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate, reference, max_n=4):
    c, r = candidate.lower().split(), reference.lower().split()
    log_sum = 0.0
    for n in range(1, max_n + 1):
        cand, ref = ngrams(c, n), ngrams(r, n)
        total = sum(cand.values())
        clipped = sum(min(count, ref[g]) for g, count in cand.items())
        if total == 0 or clipped == 0:
            return 0.0
        log_sum += math.log(clipped / total) / max_n
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
    return bp * math.exp(log_sum)


def lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x == y else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(candidate, reference):
    c, r = candidate.lower().split(), reference.lower().split()
    lcs = lcs_length(c, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(c), lcs / len(r)
    return 2 * precision * recall / (precision + recall)
