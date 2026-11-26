import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The parts of GPT-2 small you need:
#   t = model.transformer
#   t.wte   token embedding table,     shape (50257, 768)
#   t.wpe   position embedding table,  shape (1024, 768)
#   t.h     a list of 12 Transformer blocks
#   t.ln_f  the final LayerNorm
#   model.lm_head   the "unembedding": 768 numbers -> 50257 scores (logits)


def run_block(block, x):
    """Call one Transformer block. Works across transformers versions."""
    out = block(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def manual_logits(model, input_ids):
    """Rebuild model(input_ids).logits step by step.

    input_ids: long tensor (1, T).  Returns float tensor (1, T, 50257).
    """
    t = model.transformer
    # TODO 1: look up the token embeddings and the position embeddings
    #         (positions are 0, 1, ..., T-1) and add them           -> (1, T, 768)
    # TODO 2: pass the result through the 12 blocks, one after another
    # TODO 3: apply the final LayerNorm, then the unembedding        -> (1, T, 50257)
    raise NotImplementedError


@torch.no_grad()
def next_token_top_k(model, tokenizer, text, k=5):
    """The k most likely next tokens after `text`.

    Returns a list of (token_string, probability) pairs, most likely first.
    """
    # TODO 4: tokenise, get the logits, keep only the LAST position
    # TODO 5: softmax -> probabilities, take the top k, decode each token id
    raise NotImplementedError
