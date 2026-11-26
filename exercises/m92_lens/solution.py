import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gpt2():
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


@torch.no_grad()
def residual_stream(model, ids):
    """The residual stream at the last position after the embedding and after each block.

    ids: LongTensor (T,). Returns a tensor (n_layer + 1, d_model). Row 0 is token embedding plus
    position embedding, row l is the output of block l - 1, before the final LayerNorm.
    """
    states = []
    handles = [model.transformer.drop.register_forward_hook(lambda m, a, out: states.append(out[0, -1]))]
    for block in model.transformer.h:
        handles.append(block.register_forward_hook(
            lambda m, a, out: states.append((out[0] if isinstance(out, tuple) else out)[0, -1])))
    try:
        model(ids[None])
    finally:
        for h in handles:
            h.remove()
    return torch.stack(states)


@torch.no_grad()
def logit_lens(model, resid):
    """Decode every row of resid (n, d_model) with the final LayerNorm and the unembedding. Returns (n, V)."""
    return model.lm_head(model.transformer.ln_f(resid))


def token_rank(logits, token_id):
    """Rank of a token in each row of logits (n, V): 1 means it is the top prediction. Returns a list."""
    target = logits[:, token_id][:, None]
    return ((logits > target).sum(-1) + 1).tolist()


def kl_to_final(logits):
    """KL(final || layer) in nats for every row, where final is the distribution of the last row."""
    logp = torch.log_softmax(logits, -1)
    final = logp[-1]
    return (final.exp() * (final - logp)).sum(-1).tolist()
