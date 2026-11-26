import torch

# Do not call torch.logsumexp, torch.softmax, torch.log_softmax or their
# F.* / tensor-method versions. The tests disable them while your code runs.


def logsumexp(z, dim=-1, keepdim=False):
    """log(sum(exp(z))) along `dim`, without overflow."""
    # TODO 1: subtract the maximum before exponentiating, add it back afterwards
    raise NotImplementedError


def log_softmax(z, dim=-1):
    # TODO 2: use logsumexp
    raise NotImplementedError


def softmax(z, dim=-1):
    # TODO 3: use log_softmax
    raise NotImplementedError
