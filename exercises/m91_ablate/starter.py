import contextlib
import torch


@contextlib.contextmanager
def ablate_heads(model, heads):
    """Zero the output of the given heads, a list of (layer, head), while the block is active.

    In GPT-2 the outputs of the 12 heads of a layer are concatenated into one vector of size 768,
    head h occupying the slice [h * 64, (h + 1) * 64), and then passed through the linear layer
    attn.c_proj. Zeroing that slice of the input of c_proj removes the head.
    """
    n_head = model.config.n_head
    d_head = model.config.n_embd // n_head
    handles = []
    # TODO 1: group the heads by layer
    # TODO 2: for every layer write a function hook(module, args) that clones args[0], zeroes the
    #         slices of the heads and returns the new input as a tuple (x,)
    # TODO 3: register it with model.transformer.h[l].attn.c_proj.register_forward_pre_hook(hook)
    #         and keep the returned handle in handles
    raise NotImplementedError
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()          # without this the model would stay modified


@torch.no_grad()
def token_losses(model, ids):
    """Cross-entropy in nats of every next-token prediction. ids (T,) -> (T - 1,)."""
    # TODO 4: log_softmax of the logits, then pick the log-probability of the token that follows
    raise NotImplementedError


def second_half_loss(model, ids, n):
    """Mean loss on the second copy of a sequence BOS, x_1 .. x_n, x_1 .. x_n."""
    # TODO 5: predictions 0 .. n-1 are for the first copy, n .. 2n-1 for the second
    raise NotImplementedError
