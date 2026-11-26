import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def relabel(demos, mapping):
    return [(x, mapping[y]) for x, y in demos]


def build_prompt(demos, x, template="Review: {x}\nSentiment:"):
    parts = [template.format(x=dx) + " " + dy for dx, dy in demos]
    parts.append(template.format(x=x))
    return "\n\n".join(parts)


@torch.no_grad()
def classify(model, tokenizer, prompt, labels):
    logits = model(tokenizer(prompt, return_tensors="pt").input_ids).logits[0, -1]
    scores = [logits[tokenizer(" " + label).input_ids[0]].item() for label in labels]
    return labels[scores.index(max(scores))]


def follow_rate(model, tokenizer, demos, test, mapping):
    shown = relabel(demos, mapping)
    labels = list(mapping.values())
    hits = sum(classify(model, tokenizer, build_prompt(shown, x), labels) == mapping[y] for x, y in test)
    return hits / len(test)
