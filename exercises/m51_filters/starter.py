import torch

# Every function takes the logits of one position, a float tensor (V,), and returns new logits (V,).
# A token is removed by setting its logit to -inf, so that softmax gives it probability 0.


def apply_temperature(logits, temperature):
    """Divide the logits by the temperature (assumed > 0)."""
    # TODO 1
    raise NotImplementedError


def top_k_filter(logits, k):
    """Keep the k largest logits and set all others to -inf."""
    # TODO 2: torch.topk(logits, k).values[-1] is the k-th largest logit
    raise NotImplementedError


def top_p_filter(logits, p):
    """Nucleus sampling: keep the smallest set of top tokens whose total probability is at least p.

    Sort the tokens by probability, largest first. A token is kept if the probability of all tokens
    ranked above it is still below p. The top token is therefore always kept.
    """
    # TODO 3: torch.sort(logits, descending=True) gives the sorted logits and the original indices
    # TODO 4: softmax, cumulative sum, decide which sorted positions to remove
    # TODO 5: put the result back in the original order
    raise NotImplementedError


def sample_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """The usual order of operations: temperature, then top-k, then top-p, then draw one token.

    temperature == 0 means greedy decoding: return logits.argmax(-1) directly.
    Otherwise return a 0-dim long tensor drawn with torch.multinomial(..., generator=generator).
    """
    # TODO 6
    raise NotImplementedError
