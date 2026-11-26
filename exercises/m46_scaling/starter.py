import math


def fit_power_law(xs, ys):
    """Fit y = c * x^p by least squares in log-log space. Returns (c, p)."""
    # TODO 1: linear regression of log y on log x; the slope is p, exp(intercept) is c
    raise NotImplementedError


def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """The parametric loss fitted by Hoffmann et al. (2022):  L(N, D) = E + A / N^alpha + B / D^beta

    N: number of parameters, D: number of training tokens.
    """
    # TODO 2
    raise NotImplementedError


def flops(N, D):
    """Training compute in floating-point operations, with the usual approximation C = 6 N D."""
    # TODO 3
    raise NotImplementedError


def optimal_allocation(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """The (N, D) that minimises the loss for a compute budget C = 6 N D.

    N_opt = G * (C / 6)^(beta / (alpha + beta))
    D_opt = (C / 6)^(alpha / (alpha + beta)) / G
    G     = (alpha * A / (beta * B))^(1 / (alpha + beta))
    Returns (N_opt, D_opt) as floats.
    """
    # TODO 4
    raise NotImplementedError
