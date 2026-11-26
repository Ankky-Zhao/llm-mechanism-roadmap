import torch

# No autograd in this exercise. The tests call your functions inside
# torch.no_grad(), so .backward() will not work. Write the formulas.


def softmax_jacobian(z):
    """z: (V,). Returns J of shape (V, V) with J[i, j] = d p_i / d z_j."""
    # TODO 1: J = diag(p) - p p^T
    raise NotImplementedError


def ce_grad_logits(logits, targets):
    """Gradient of the MEAN cross-entropy with respect to the logits.

    logits: (N, V), targets: (N,) long. Returns (N, V).
    """
    # TODO 2: (p - onehot(targets)) / N
    raise NotImplementedError


def linear_ce_backward(x, W, b, targets):
    """One linear layer followed by softmax cross-entropy.

    logits = x @ W.T + b     with x: (N, D), W: (V, D), b: (V,)
    Returns (loss, dW, db, dx) for the mean cross-entropy.
    """
    # TODO 3: forward pass and loss
    # TODO 4: dlogits from ce_grad_logits, then the chain rule for W, b and x
    raise NotImplementedError
