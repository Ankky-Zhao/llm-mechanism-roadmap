import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Does a model learn the mapping shown in the examples, or does it answer from what it already knows?
# The examples are relabelled through a mapping, for instance {"positive": "negative", "negative": "positive"},
# and the test measures how often the model's answer follows that mapping.


def relabel(demos, mapping):
    """Replace every label y in the list of (x, y) pairs by mapping[y]."""
    # TODO 1
    raise NotImplementedError


def build_prompt(demos, x, template="Review: {x}\nSentiment:"):
    """As in exercise 1. Nothing to change here."""
    parts = [template.format(x=dx) + " " + dy for dx, dy in demos]
    parts.append(template.format(x=x))
    return "\n\n".join(parts)


@torch.no_grad()
def classify(model, tokenizer, prompt, labels):
    """As in exercise 1: the label whose first token (with a leading space) has the highest logit."""
    # TODO 2
    raise NotImplementedError


def follow_rate(model, tokenizer, demos, test, mapping):
    """How often does the model answer according to the mapping shown in the examples?

    demos, test: lists of (review, true_label) with true labels "positive" / "negative".
    Show the demos relabelled through `mapping`, let the model choose among mapping.values(),
    and count an answer as following if it equals mapping[true_label].
    Returns a fraction between 0 and 1. A value of 0 means the model always gives the other label.
    """
    # TODO 3
    raise NotImplementedError
