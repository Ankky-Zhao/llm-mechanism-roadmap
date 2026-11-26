import torch

TEMPLATE = "The city of {} is in"


def prompt(tok, city):
    """Returns (ids (1, T), position of the last token of the city name)."""
    ids = tok(TEMPLATE.format(city), return_tensors="pt").input_ids
    pos = len(tok("The city of " + city).input_ids) - 1
    return ids, pos


@torch.no_grad()
def city_state(model, tok, city, layer):
    """Output of block `layer` at the last token of the city name. Returns (768,)."""
    ids, pos = prompt(tok, city)
    return model(ids, output_hidden_states=True).hidden_states[layer + 1][0, pos]


def mean_difference(model, tok, cities_to, cities_from, layer):
    """Mean state of cities_to minus mean state of cities_from."""
    a = torch.stack([city_state(model, tok, c, layer) for c in cities_to]).mean(0)
    b = torch.stack([city_state(model, tok, c, layer) for c in cities_from]).mean(0)
    return a - b


@torch.no_grad()
def steered_logit_diff(model, tok, city, layer, vec, token_to, token_from):
    """logit(token_to) - logit(token_from) for the next token, with vec added to the output of block
    `layer` at the last token of the city name. vec None means no intervention."""
    ids, pos = prompt(tok, city)
    handle = None
    if vec is not None:
        def hook(module, args, out):
            x = (out[0] if isinstance(out, tuple) else out).clone()
            x[0, pos] += vec
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        logits = model(ids).logits[0, -1]
    finally:
        if handle is not None:
            handle.remove()
    return (logits[token_to] - logits[token_from]).item()
