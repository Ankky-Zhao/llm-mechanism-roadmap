import torch


def run_with_grad(model, ids, ans, wrong):
    """Run on ids and return (activations, gradients) of the mean logit difference.

    Both are dicts with the keys ("z", l) and ("resid", l) as in run_with_cache, the gradients having
    the same shapes as the activations.
    """
    acts, handles = {}, []

    def keep(key, t):
        t.retain_grad()
        acts[key] = t
    for l, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(lambda m, a, out, l=l: keep(("resid", l), block_output(out))))
        handles.append(block.attn.c_proj.register_forward_pre_hook(lambda m, a, l=l: keep(("z", l), a[0])))
    try:
        metric = logit_diff(model(ids).logits, ans, wrong).mean()
        model.zero_grad()
        metric.backward()
    finally:
        for h in handles:
            h.remove()
    grads = {k: v.grad.detach() for k, v in acts.items()}
    return {k: v.detach() for k, v in acts.items()}, grads


def attribution_heads(model, clean_cache, corrupt_acts, corrupt_grads, scale):
    """First-order estimate of the effect of patching every head at all positions.

    estimate = sum over batch, positions and the 64 dimensions of (clean - corrupt) * gradient.
    The gradient belongs to the mean over the batch, so the sum over the batch gives the change of
    the mean. scale is mean clean logit difference minus mean corrupted one. Returns (n_layer, n_head).
    """
    L, H = model.config.n_layer, model.config.n_head
    out = torch.zeros(L, H)
    for l in range(L):
        d = (clean_cache[("z", l)] - corrupt_acts[("z", l)]) * corrupt_grads[("z", l)]
        N, T, D = d.shape
        out[l] = d.reshape(N, T, H, D // H).sum((0, 1, 3)) / scale
    return out


def attribution_residual(model, clean_cache, corrupt_acts, corrupt_grads, scale):
    """The same estimate for the residual stream at every (layer, position). Returns (n_layer, T)."""
    L = model.config.n_layer
    rows = []
    for l in range(L):
        d = (clean_cache[("resid", l)] - corrupt_acts[("resid", l)]) * corrupt_grads[("resid", l)]
        rows.append(d.sum((0, 2)) / scale)
    return torch.stack(rows)
