import torch


def causal_mask(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def masked_softmax(scores, mask):
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)


@torch.no_grad()
def future_leak(f, x, pos):
    x2 = x.clone()
    if x.is_floating_point():
        x2[pos:] = torch.randn_like(x[pos:])
    else:
        x2[pos:] = torch.randint_like(x[pos:], 50257)
    return (f(x)[:pos] - f(x2)[:pos]).abs().max().item()
