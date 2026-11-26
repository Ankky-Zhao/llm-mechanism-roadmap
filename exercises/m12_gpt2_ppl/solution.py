import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def token_nlls(model, input_ids):
    logits = model(input_ids).logits                  # (1, T, V)
    pred = logits[0, :-1]                             # positions 0..T-2 predict tokens 1..T-1
    gold = input_ids[0, 1:]                           # (T-1,)
    log_probs = torch.log_softmax(pred, dim=-1)
    return -log_probs[torch.arange(gold.shape[0]), gold]


def text_perplexity(model, tokenizer, text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    return torch.exp(token_nlls(model, input_ids).mean()).item()
