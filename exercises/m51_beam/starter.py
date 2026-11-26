import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def sequence_logprob(model, input_ids, prompt_len):
    """Total log-probability of the continuation, log p(x_{prompt_len} ... x_{T-1} | prompt).

    input_ids: (1, T). Returns a Python float.
    """
    # TODO 1: log-softmax of model(input_ids).logits[0, :-1]; row t predicts token t+1
    # TODO 2: pick the log-probability of each actual next token, sum the ones after the prompt
    raise NotImplementedError


@torch.no_grad()
def beam_search(model, input_ids, beam_size, max_new_tokens):
    """Beam search without end-of-sequence handling: every beam grows by exactly max_new_tokens.

    input_ids: (1, T)
    returns (beams, scores): beams (beam_size, T + max_new_tokens), best first;
                             scores (beam_size,), the total log-probability of each continuation.

    Keep `beams` (b, t) and `scores` (b,). Start with the prompt alone and score 0. In each step:
      logp  = log_softmax(model(beams).logits[:, -1])        (b, V)
      total = scores[:, None] + logp                         every beam extended by every token
      take the beam_size largest entries of total.view(-1); entry i belongs to
      beam i // V and token i % V
    """
    # TODO 3
    raise NotImplementedError
