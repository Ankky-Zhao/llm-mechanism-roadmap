import torch


@torch.no_grad()
def patch_head(model, corrupt_ids, clean_cache, layer, head, pos=slice(None)):
    """Run on the corrupted input with the output of one head replaced by its clean value.

    The head output is the slice [head * 64, (head + 1) * 64) of the input of attn.c_proj.
    pos is an int or slice(None). Returns the logits (N, T, V).
    """
    d = model.config.n_embd // model.config.n_head
    sl = slice(head * d, (head + 1) * d)

    def hook(module, args):
        z = args[0].clone()
        z[:, pos, sl] = clean_cache[("z", layer)][:, pos, sl]
        return (z,)
    handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook)
    try:
        return model(corrupt_ids).logits
    finally:
        handle.remove()


def head_patching_map(model, clean_ids, corrupt_ids, ans, wrong, pos=slice(None)):
    """Normalized effect of patching every head. Returns a tensor (n_layer, n_head)."""
    clean_logits, cache = run_with_cache(model, clean_ids)
    with torch.no_grad():
        corrupt_logits = model(corrupt_ids).logits
    c, x = logit_diff(clean_logits, ans, wrong), logit_diff(corrupt_logits, ans, wrong)
    L, H = model.config.n_layer, model.config.n_head
    out = torch.zeros(L, H)
    for l in range(L):
        for h in range(H):
            out[l, h] = normalized(logit_diff(patch_head(model, corrupt_ids, cache, l, h, pos), ans, wrong), c, x)
    return out
