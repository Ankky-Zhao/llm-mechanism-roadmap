import math
import torch


def _scores(d_k, T, n, scaled, generator):
    q = torch.randn(n, 1, d_k, generator=generator)
    k = torch.randn(n, T, d_k, generator=generator)
    s = (q * k).sum(-1)                               # (n, T)
    return s / math.sqrt(d_k) if scaled else s


def dot_product_variance(d_k, n=20000, scaled=False, generator=None):
    q = torch.randn(n, d_k, generator=generator)
    k = torch.randn(n, d_k, generator=generator)
    dots = (q * k).sum(-1)
    if scaled:
        dots = dots / math.sqrt(d_k)
    return dots.var().item()


def mean_max_weight(d_k, T=16, n=2000, scaled=False, generator=None):
    weights = torch.softmax(_scores(d_k, T, n, scaled, generator), dim=-1)
    return weights.max(dim=-1).values.mean().item()


def score_gradient_norm(d_k, T=16, n=500, scaled=False, generator=None):
    scores = _scores(d_k, T, n, scaled, generator).detach().requires_grad_(True)
    r = torch.randn(n, T, generator=generator)
    (torch.softmax(scores, dim=-1) * r).sum().backward()
    return scores.grad.norm(dim=-1).mean().item()
