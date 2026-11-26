import torch


def cross_entropy(logits, targets):
    """Mean negative log-likelihood in nats.

    logits:  float tensor (N, V), unnormalised scores
    targets: long tensor (N,), the index of the correct token
    Do not call F.cross_entropy, F.nll_loss or F.log_softmax.
    torch.logsumexp is allowed here (you will write it yourself in module 1.3).
    """
    # TODO 1: log-probabilities  log p = logits - logsumexp(logits)
    # TODO 2: pick log p of the correct token for every row
    # TODO 3: return the mean of the negative values
    raise NotImplementedError


def perplexity(logits, targets):
    """Perplexity = exp(mean NLL). Returns a 0-dim tensor."""
    # TODO 4
    raise NotImplementedError


def bits_per_token(logits, targets):
    """The same loss measured in bits instead of nats."""
    # TODO 5
    raise NotImplementedError
