import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sample_next(logits, temperature=1.0, generator=None):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)[0]


@torch.no_grad()
def sample_generate(model, input_ids, max_new_tokens=20, temperature=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[0, -1]
        next_id = sample_next(logits, temperature, generator=g)
        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
    return input_ids
