import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# A natural language inference (NLI) model reads a premise and a hypothesis and decides whether the
# premise entails the hypothesis, contradicts it, or neither. Used with a retrieved passage as premise
# and a sentence of the answer as hypothesis, it checks whether the answer is supported by the passage.


@torch.no_grad()
def entailment_prob(model, tokenizer, premise, hypothesis):
    """Probability that the premise entails the hypothesis, as a float.

    Tokenise the pair with tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).
    The model has three output classes; model.config.id2label says which index is "entailment".
    Do not hard-code the index: it differs between models.
    """
    # TODO 1
    raise NotImplementedError


def split_sentences(text):
    """Split a text after ., ! or ? followed by whitespace. Strip and drop empty pieces."""
    # TODO 2: re.split(r"(?<=[.!?])\\s+", text.strip())
    raise NotImplementedError


def groundedness(model, tokenizer, answer, passages):
    """Is every sentence of the answer supported by at least one passage?

    For each sentence take the highest entailment probability over the passages.
    Returns (score, per_sentence): score is the LOWEST of these values, since one unsupported
    sentence is enough to make an answer ungrounded; per_sentence is the list of values.
    """
    # TODO 3
    raise NotImplementedError
