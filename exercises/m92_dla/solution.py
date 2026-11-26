import torch


@torch.no_grad()
def components(model, ids):
    """The terms whose sum is the residual stream at the last position before the final LayerNorm.

    Returns (names, parts) with parts of shape (1 + 2 * n_layer, d_model): the embedding, then for
    each layer the output of its attention sublayer and of its MLP.
    """
    names, parts = ["embed"], []
    handles = [model.transformer.drop.register_forward_hook(lambda m, a, out: parts.append(out[0, -1]))]
    for l, block in enumerate(model.transformer.h):
        names += [f"attn{l}", f"mlp{l}"]
        handles.append(block.attn.register_forward_hook(
            lambda m, a, out: parts.append((out[0] if isinstance(out, tuple) else out)[0, -1])))
        handles.append(block.mlp.register_forward_hook(lambda m, a, out: parts.append(out[0, -1])))
    try:
        model(ids[None])
    finally:
        for h in handles:
            h.remove()
    return names, torch.stack(parts)


@torch.no_grad()
def direct_logit_attribution(model, parts, token_a, token_b):
    """Contribution of every part to logit(token_a) - logit(token_b).

    The final LayerNorm computes g * (x - mean(x)) / std(x) + b. With std(x) of the full residual
    stream held fixed, this is linear in x, so the logit difference splits into one term per part
    plus a constant that comes from b. Returns (contributions (n_parts,), constant).
    """
    ln = model.transformer.ln_f
    total = parts.sum(0)
    centered_total = total - total.mean()
    std = torch.sqrt(centered_total.pow(2).mean() + ln.eps)
    direction = model.lm_head.weight[token_a] - model.lm_head.weight[token_b]
    centered = parts - parts.mean(-1, keepdim=True)
    contrib = (centered / std * ln.weight) @ direction
    const = (ln.bias @ direction).item()
    return contrib, const
