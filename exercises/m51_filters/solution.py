import torch


def apply_temperature(logits, temperature):
    return logits / temperature


def top_k_filter(logits, k):
    kth = torch.topk(logits, k).values[..., -1, None]            # the k-th largest logit
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    sorted_logits, order = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    mass_before = probs.cumsum(-1) - probs                       # probability of all tokens ranked above this one
    sorted_logits = sorted_logits.masked_fill(mass_before >= p, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)


def sample_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0:
        return logits.argmax(-1)
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    return torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)[0]
