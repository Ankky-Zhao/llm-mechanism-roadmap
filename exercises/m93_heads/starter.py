import torch


@torch.no_grad()
def patch_head(model, corrupt_ids, clean_cache, layer, head, pos=slice(None)):
    """Run on the corrupted input with the output of one head replaced by its clean value.

    The head output is the slice [head * 64, (head + 1) * 64) of the input of attn.c_proj.
    pos is an int or slice(None). Returns the logits (N, T, V).
    """
    d = model.config.n_embd // model.config.n_head
    sl = slice(head * d, (head + 1) * d)
    # TODO 1: a forward pre-hook on model.transformer.h[layer].attn.c_proj that clones args[0],
    #         overwrites z[:, pos, sl] with clean_cache[("z", layer)][:, pos, sl] and returns (z,)
    # TODO 2: run on corrupt_ids, remove the handle, return the logits
    raise NotImplementedError


def head_patching_map(model, clean_ids, corrupt_ids, ans, wrong, pos=slice(None)):
    """Normalized effect of patching every head. Returns a tensor (n_layer, n_head)."""
    # TODO 3: as residual_patching_map, with the loop over layers and heads
    raise NotImplementedError
