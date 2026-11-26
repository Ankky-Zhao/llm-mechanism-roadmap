import itertools
import re
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The idea of SelfCheckGPT: ask the same question several times with sampling. What a model knows
# it tends to say the same way every time; what it makes up tends to come out differently each time.


@torch.no_grad()
def sample_answers(model, tokenizer, question, n=5, temperature=1.0, seed=0, max_new_tokens=12):
    """n sampled short answers to one question, as a list of strings.

    Call torch.manual_seed(seed) first. The user message is question + " Answer with a few words only."
    Generate with do_sample=True, temperature=temperature, top_p=1.0, top_k=0, num_return_sequences=n.
    Decode only the new tokens of every sequence (skip_special_tokens=True) and strip whitespace.
    """
    # TODO 1
    raise NotImplementedError


def overlap(a, b):
    """Token F1 between two answers: lower case, keep only letters, digits and spaces, split on whitespace.
    Shared tokens count with multiplicity. Two empty answers give 1.0, one empty answer gives 0.0."""
    # TODO 2
    raise NotImplementedError


def consistency(answers):
    """The mean overlap over all unordered pairs of answers (itertools.combinations)."""
    # TODO 3
    raise NotImplementedError
