import math


def fit_power_law(xs, ys):
    u = [math.log(x) for x in xs]
    v = [math.log(y) for y in ys]
    mu, mv = sum(u) / len(u), sum(v) / len(v)
    p = sum((a - mu) * (b - mv) for a, b in zip(u, v)) / sum((a - mu) ** 2 for a in u)
    return math.exp(mv - p * mu), p


def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    return E + A / N ** alpha + B / D ** beta


def flops(N, D):
    return 6 * N * D


def optimal_allocation(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    D = (C / 6) ** (alpha / (alpha + beta)) / G
    return N, D
