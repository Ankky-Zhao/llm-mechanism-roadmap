import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def top2_gap(logits):
    top = torch.topk(logits.float(), 2).values
    return (top[0] - top[1]).item()


@torch.no_grad()
def greedy_trace(model, input_ids, max_new_tokens):
    tokens, gaps = [], []
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[0, -1]
        gaps.append(top2_gap(logits))
        next_id = logits.argmax()
        tokens.append(next_id.item())
        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
    return tokens, gaps


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None
