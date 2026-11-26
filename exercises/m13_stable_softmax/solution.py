import torch


def logsumexp(z, dim=-1, keepdim=False):
    m = z.max(dim=dim, keepdim=True).values           # largest entry, so exp(z - m) <= 1
    out = m + torch.log(torch.exp(z - m).sum(dim=dim, keepdim=True))
    return out if keepdim else out.squeeze(dim)


def log_softmax(z, dim=-1):
    return z - logsumexp(z, dim=dim, keepdim=True)


def softmax(z, dim=-1):
    return torch.exp(log_softmax(z, dim=dim))
