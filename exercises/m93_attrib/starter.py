import torch


def run_with_grad(model, ids, ans, wrong):
    """Run on ids and return (activations, gradients) of the mean logit difference.

    Both are dicts with the keys ("z", l) and ("resid", l) as in run_with_cache, the gradients having
    the same shapes as the activations.
    """
    acts, handles = {}, []

    def keep(key, t):
        t.retain_grad()          # ask autograd to keep the gradient of this intermediate tensor
        acts[key] = t
    # TODO 1: register the same two hooks per block as in run_with_cache, calling keep(...) on the
    #         tensors themselves (no clone, no torch.no_grad)
    # TODO 2: metric = mean logit difference of model(ids).logits, then metric.backward(), remove the handles
    # TODO 3: return ({key: activation.detach()}, {key: activation.grad.detach()})
    raise NotImplementedError


def attribution_heads(model, clean_cache, corrupt_acts, corrupt_grads, scale):
    """First-order estimate of the effect of patching every head at all positions.

    estimate = sum over batch, positions and the 64 dimensions of (clean - corrupt) * gradient.
    The gradient belongs to the mean over the batch, so the sum over the batch gives the change of
    the mean. scale is mean clean logit difference minus mean corrupted one. Returns (n_layer, n_head).
    """
    # TODO 4: for every layer form the product, reshape (N, T, 768) to (N, T, 12, 64) and sum over
    #         all axes except the head axis, then divide by scale
    raise NotImplementedError


def attribution_residual(model, clean_cache, corrupt_acts, corrupt_grads, scale):
    """The same estimate for the residual stream at every (layer, position). Returns (n_layer, T)."""
    # TODO 5: the product for ("resid", l), summed over the batch and the 768 dimensions
    raise NotImplementedError
