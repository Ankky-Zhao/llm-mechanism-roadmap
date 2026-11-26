import math
import time
import torch


def layer_multiplications(T, d):
    projections = 4 * T * d * d          # each of T positions goes through four d x d matrices
    scores = T * T * d                   # h heads, each T x T dot products of size d/h
    mixing = T * T * d                   # h heads, each a (T x T) @ (T x d/h) product
    mlp = 2 * T * d * 4 * d              # up to 4d and back down
    return {"projections": projections, "scores": scores, "mixing": mixing, "mlp": mlp,
            "linear": projections + mlp, "quadratic": scores + mixing}


def crossover_length(d):
    return 6 * d                         # 2 T^2 d = 12 T d^2


def attention_matrix_bytes(T, n_head, n_layer, bytes_per_number=2, batch=1):
    return batch * n_layer * n_head * T * T * bytes_per_number


def fit_exponent(xs, ys):
    u = [math.log(x) for x in xs]
    v = [math.log(y) for y in ys]
    mu, mv = sum(u) / len(u), sum(v) / len(v)
    return sum((a - mu) * (b - mv) for a, b in zip(u, v)) / sum((a - mu) ** 2 for a in u)


def time_attention(T, d=64, repeats=3):
    q, k, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        torch.softmax(q @ k.T / math.sqrt(d), dim=-1) @ v
        best = min(best, time.perf_counter() - t0)
    return best
