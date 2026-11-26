import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def token_nlls(model, input_ids):
    """Negative log-likelihood of every token given the tokens before it.

    input_ids: long tensor (1, T)
    returns:   float tensor (T-1,), entry t-1 is  -log p(x_t | x_<t)  for t = 1..T-1
    The first token has no context, so it gets no score.
    """
    # TODO 1: logits = model(input_ids).logits            -> (1, T, V)
    # TODO 2: the logits at position t predict token t+1. Align the two.
    # TODO 3: log-softmax over the vocabulary, pick the true next token, negate
    raise NotImplementedError


def text_perplexity(model, tokenizer, text):
    """Perplexity of `text` under `model`, as a Python float."""
    # TODO 4: tokenise, call token_nlls, return exp(mean)
    raise NotImplementedError
