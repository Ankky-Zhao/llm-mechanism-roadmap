import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def greedy_generate(model, input_ids, max_new_tokens=20):
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[0, -1]            # scores for the next token
        next_id = logits.argmax()                          # softmax is not needed: it keeps the order
        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
    return input_ids


def repeated_ngrams(token_ids, n=4):
    seen, repeats = set(), 0
    for i in range(len(token_ids) - n + 1):
        gram = tuple(token_ids[i:i + n])
        repeats += gram in seen
        seen.add(gram)
    return repeats
