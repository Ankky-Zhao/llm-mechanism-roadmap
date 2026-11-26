import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The embedding model of exercise 1 is a bi-encoder: question and passage are encoded separately and
# compared by a dot product, so passage vectors can be computed once and stored. A cross-encoder reads
# question and passage TOGETHER and outputs one relevance score. It is more accurate and far slower,
# so it is applied only to the best few candidates of a first, cheap search.


@torch.no_grad()
def cross_scores(model, tokenizer, query, passages):
    """Relevance score of every passage for the query, as a list of floats.

    Tokenise the pairs with tokenizer([query] * n, passages, padding=True, truncation=True, return_tensors="pt").
    The model returns logits of shape (n, 1); the score is that single number.
    """
    # TODO 1
    raise NotImplementedError


def rerank(model, tokenizer, query, corpus, candidates, top_n=8):
    """Re-order the first top_n candidates (document indices) by cross-encoder score, best first.
    Candidates after position top_n keep their order and follow unchanged."""
    # TODO 2
    raise NotImplementedError
