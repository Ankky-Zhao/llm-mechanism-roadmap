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
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    def make_hook(hs):
        def hook(module, args):
            x = args[0].clone()
            for h in hs:
                x[..., h * d_head:(h + 1) * d_head] = 0
            return (x,)
        return hook

    handles = [model.transformer.h[l].attn.c_proj.register_forward_pre_hook(make_hook(hs)) for l, hs in by_layer.items()]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


@torch.no_grad()
def token_losses(model, ids):
    """Cross-entropy in nats of every next-token prediction. ids (T,) -> (T - 1,)."""
    logp = torch.log_softmax(model(ids[None]).logits[0], -1)
    return -logp[:-1].gather(1, ids[1:, None])[:, 0]


def second_half_loss(model, ids, n):
    """Mean loss on the second copy of a sequence BOS, x_1 .. x_n, x_1 .. x_n."""
    return token_losses(model, ids)[n:].mean().item()
