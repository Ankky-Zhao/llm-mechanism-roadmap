import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def greedy_generate(model, input_ids, max_new_tokens=20):
    """Greedy decoding: at every step append the single most likely token.

    input_ids: long tensor (1, T).  Returns long tensor (1, T + max_new_tokens).
    Do not call model.generate.
    """
    for _ in range(max_new_tokens):
        # TODO 1: logits = model(input_ids).logits, keep the last position   -> (50257,)
        # TODO 2: next_id = the index of the largest logit
        # TODO 3: append next_id to input_ids (torch.cat along dim 1)
        raise NotImplementedError
    return input_ids


def repeated_ngrams(token_ids, n=4):
    """How many n-grams in the list `token_ids` have already appeared earlier in it."""
    # TODO 4: walk over all windows of length n, count the ones seen before
    raise NotImplementedError
