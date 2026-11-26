import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
         "Karen", "Leo", "Maria", "Nick", "Olivia", "Paul", "Quinn", "Rachel", "Sam", "Tina"]


def build_prompt(names, codes, i):
    """A list of facts followed by a question about fact number i. Nothing to change."""
    context = "".join(f"The code for {n} is {c}.\n" for n, c in zip(names, codes))
    return context + f"The code for {names[i]} is"


@torch.no_grad()
def retrieval_correct(model, tok, names, codes, i):
    """Does the model pick the right code for names[i]?

    Build the prompt, take the logits of the last position, and compare ONLY the candidate
    tokens: the first token of ' ' + str(code) for every code in the list.
    Return True if the candidate with the largest logit belongs to codes[i].
    """
    # TODO 1
    raise NotImplementedError


def accuracy_by_position(model, tok, n_items=5, trials=12, seed=0):
    """Accuracy as a function of where the queried fact sits in the context.

    For trial t use  r = random.Random(seed + t),  names = r.sample(NAMES, n_items),
    codes = r.sample(range(10, 100), n_items)  (in this order), and query every position.
    Returns a list of n_items floats: the share of trials in which position i was answered correctly.
    """
    # TODO 2
    raise NotImplementedError
