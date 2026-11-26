import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def sequence_logprob(model, input_ids, prompt_len):
    logp = torch.log_softmax(model(input_ids).logits[0, :-1], -1)             # row t predicts token t+1
    picked = logp.gather(-1, input_ids[0, 1:, None])[:, 0]
    return picked[prompt_len - 1:].sum().item()


@torch.no_grad()
def beam_search(model, input_ids, beam_size, max_new_tokens):
    beams = input_ids                                                         # (1, T) at the start
    scores = torch.zeros(1)
    for _ in range(max_new_tokens):
        logp = torch.log_softmax(model(beams).logits[:, -1], -1)              # (b, V)
        V = logp.shape[-1]
        total = (scores[:, None] + logp).view(-1)                             # every beam extended by every token
        scores, flat = total.topk(beam_size)
        parent, token = flat // V, flat % V
        beams = torch.cat([beams[parent], token[:, None]], dim=1)
    return beams, scores
