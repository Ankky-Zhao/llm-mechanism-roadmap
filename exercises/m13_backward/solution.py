import torch


def softmax_jacobian(z):
    p = torch.softmax(z, dim=-1)
    return torch.diag(p) - torch.outer(p, p)          # p_i (delta_ij - p_j)


def ce_grad_logits(logits, targets):
    N = logits.shape[0]
    g = torch.softmax(logits, dim=-1)                 # p
    g[torch.arange(N), targets] -= 1.0                # p - y
    return g / N                                      # mean over the batch


def linear_ce_backward(x, W, b, targets):
    N = x.shape[0]
    logits = x @ W.T + b                              # (N, V)
    log_p = torch.log_softmax(logits, dim=-1)
    loss = -log_p[torch.arange(N), targets].mean()
    dlogits = ce_grad_logits(logits, targets)         # (N, V)
    dW = dlogits.T @ x                                # (V, D)
    db = dlogits.sum(dim=0)                           # (V,)
    dx = dlogits @ W                                  # (N, D)
    return loss, dW, db, dx
