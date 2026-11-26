import math
import torch


def cross_entropy(logits, targets):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)   # (N, V)
    picked = log_probs[torch.arange(logits.shape[0]), targets]           # (N,)
    return -picked.mean()


def perplexity(logits, targets):
    return torch.exp(cross_entropy(logits, targets))


def bits_per_token(logits, targets):
    return cross_entropy(logits, targets) / math.log(2)
