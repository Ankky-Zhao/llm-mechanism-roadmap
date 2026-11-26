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
    handles = []
    # TODO 1: register a forward hook on model.transformer.drop (its output is the embedding sum)
    #         that appends out[0, -1] to states. A forward hook has the signature hook(module, args, output).
    # TODO 2: register a forward hook on every block in model.transformer.h. The output of a block
    #         can be a tuple whose first element is the hidden state.
    # TODO 3: run the model once, remove the handles, stack the states.
    #         Do not use output_hidden_states here: its last entry already has the final LayerNorm applied.
    raise NotImplementedError


@torch.no_grad()
def logit_lens(model, resid):
    """Decode every row of resid (n, d_model) with the final LayerNorm and the unembedding. Returns (n, V)."""
    # TODO 4: model.transformer.ln_f, then model.lm_head
    raise NotImplementedError


def token_rank(logits, token_id):
    """Rank of a token in each row of logits (n, V): 1 means it is the top prediction. Returns a list."""
    # TODO 5: one plus the number of tokens with a strictly larger logit
    raise NotImplementedError


def kl_to_final(logits):
    """KL(final || layer) in nats for every row, where final is the distribution of the last row."""
    # TODO 6: sum_v p_final(v) * (log p_final(v) - log p_layer(v))
    raise NotImplementedError
