import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Greedy decoding has no randomness: the same logits always give the same token.
# The logits themselves, however, are floating-point numbers, and the same model computed in a
# different precision, on different hardware or with a different batch gives slightly different logits.
# This exercise measures when such small differences change the output.


def top2_gap(logits):
    """Difference between the largest and the second largest logit, as a Python float.

    This is the margin by which greedy decoding makes its choice.
    Convert with logits.float() first so that half-precision inputs work too.
    """
    # TODO 1
    raise NotImplementedError


@torch.no_grad()
def greedy_trace(model, input_ids, max_new_tokens):
    """Greedy decoding that also records how close each decision was.

    Returns (tokens, gaps): two lists of length max_new_tokens,
    the chosen token ids (ints) and the top-2 gap of the logits at each step.
    """
    # TODO 2: the loop from module 0.1, plus top2_gap at every step
    raise NotImplementedError


def first_divergence(a, b):
    """Index of the first position at which two equally long token lists differ, or None."""
    # TODO 3
    raise NotImplementedError
