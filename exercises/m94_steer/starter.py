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
    # TODO 1: hidden_states[layer + 1] is the output of block `layer`
    raise NotImplementedError


def mean_difference(model, tok, cities_to, cities_from, layer):
    """Mean state of cities_to minus mean state of cities_from."""
    # TODO 2
    raise NotImplementedError


@torch.no_grad()
def steered_logit_diff(model, tok, city, layer, vec, token_to, token_from):
    """logit(token_to) - logit(token_from) for the next token, with vec added to the output of block
    `layer` at the last token of the city name. vec None means no intervention."""
    ids, pos = prompt(tok, city)
    # TODO 3: if vec is given, a forward hook on model.transformer.h[layer] that adds vec at [0, pos]
    #         (clone first, and return a tuple if the block output is a tuple)
    # TODO 4: run, remove the hook, return the logit difference at the last position as a float
    raise NotImplementedError
