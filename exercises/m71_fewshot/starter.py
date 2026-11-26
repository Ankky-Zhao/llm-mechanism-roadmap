from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(demos, x, template="Review: {x}\nSentiment:"):
    """A few-shot prompt.

    demos: list of (input, label) pairs. Each becomes  template.format(x=input) + " " + label
    The query x comes last, as template.format(x=x) with no label.
    The blocks are joined by a blank line ("\n\n"). With no demos the result is just the query block.
    """
    # TODO 1
    raise NotImplementedError


@torch.no_grad()
def label_scores(model, tokenizer, prompt, labels):
    """Classification without generating text: read the logits of the label words directly.

    Run the prompt through the model and take the logits at the last position. For every label,
    look up the logit of the FIRST token of " " + label (with the leading space, because the label
    follows a colon and a space). Returns a list of Python floats, one per label.
    """
    # TODO 2
    raise NotImplementedError


def classify(model, tokenizer, prompt, labels):
    """The label with the highest score."""
    # TODO 3
    raise NotImplementedError


def majority_vote(answers):
    """Self-consistency: the most frequent answer among several sampled ones.

    Ignore None (runs from which no answer could be extracted). On a tie return the answer that
    appears first in the list. Returns None if nothing is left.
    """
    # TODO 4
    raise NotImplementedError
