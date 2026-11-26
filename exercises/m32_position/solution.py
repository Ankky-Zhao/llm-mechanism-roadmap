import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
         "Karen", "Leo", "Maria", "Nick", "Olivia", "Paul", "Quinn", "Rachel", "Sam", "Tina"]


def build_prompt(names, codes, i):
    context = "".join(f"The code for {n} is {c}.\n" for n, c in zip(names, codes))
    return context + f"The code for {names[i]} is"


@torch.no_grad()
def retrieval_correct(model, tok, names, codes, i):
    ids = tok(build_prompt(names, codes, i), return_tensors="pt").input_ids
    logits = model(ids).logits[0, -1]
    candidates = torch.tensor([tok(" " + str(c)).input_ids[0] for c in codes])
    return bool(logits[candidates].argmax() == i)


def accuracy_by_position(model, tok, n_items=5, trials=12, seed=0):
    hits = [0] * n_items
    for t in range(trials):
        r = random.Random(seed + t)
        names = r.sample(NAMES, n_items)
        codes = r.sample(range(10, 100), n_items)
        for i in range(n_items):
            hits[i] += retrieval_correct(model, tok, names, codes, i)
    return [h / trials for h in hits]
