import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sample_next(logits, temperature=1.0, generator=None):
    """Draw one token id from softmax(logits / temperature).

    logits: float tensor (V,).  Returns a 0-dim long tensor.
    """
    # TODO 1: divide the logits by the temperature, then softmax
    # TODO 2: draw one index with torch.multinomial(probs, 1, generator=generator)
    raise NotImplementedError


@torch.no_grad()
def sample_generate(model, input_ids, max_new_tokens=20, temperature=1.0, seed=0):
    """Like greedy_generate, but each new token is sampled. Same seed, same output."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(max_new_tokens):
        # TODO 3: logits of the last position -> sample_next(..., generator=g) -> append
        raise NotImplementedError
    return input_ids
