import torch


@torch.no_grad()
def components(model, ids):
    """The terms whose sum is the residual stream at the last position before the final LayerNorm.

    Returns (names, parts) with parts of shape (1 + 2 * n_layer, d_model): the embedding, then for
    each layer the output of its attention sublayer and of its MLP.
    """
    names, parts = ["embed"], []
    handles = []
    # TODO 1: a forward hook on model.transformer.drop that appends out[0, -1]
    # TODO 2: for every block, hooks on block.attn (its output can be a tuple) and on block.mlp,
    #         and the names f"attn{l}", f"mlp{l}" in the same order
    # TODO 3: run the model, remove the handles, stack
    raise NotImplementedError


@torch.no_grad()
def direct_logit_attribution(model, parts, token_a, token_b):
    """Contribution of every part to logit(token_a) - logit(token_b).

    The final LayerNorm computes g * (x - mean(x)) / std(x) + b. With std(x) of the full residual
    stream held fixed, this is linear in x, so the logit difference splits into one term per part
    plus a constant that comes from b. Returns (contributions (n_parts,), constant).
    """
    ln = model.transformer.ln_f
    # TODO 4: total = parts.sum(0). std = sqrt(mean((total - mean(total))^2) + ln.eps)
    # TODO 5: direction = row token_a minus row token_b of model.lm_head.weight
    # TODO 6: centre every part by its own mean, divide by std, multiply by ln.weight, dot with direction
    # TODO 7: the constant is ln.bias dotted with direction
    raise NotImplementedError
