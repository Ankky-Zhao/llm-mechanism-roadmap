from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(demos, x, template="Review: {x}\nSentiment:"):
    parts = [template.format(x=dx) + " " + dy for dx, dy in demos]
    parts.append(template.format(x=x))
    return "\n\n".join(parts)


@torch.no_grad()
def label_scores(model, tokenizer, prompt, labels):
    logits = model(tokenizer(prompt, return_tensors="pt").input_ids).logits[0, -1]
    return [logits[tokenizer(" " + label).input_ids[0]].item() for label in labels]


def classify(model, tokenizer, prompt, labels):
    scores = label_scores(model, tokenizer, prompt, labels)
    return labels[scores.index(max(scores))]


def majority_vote(answers):
    answers = [a for a in answers if a is not None]
    if not answers:
        return None
    counts = Counter(answers)
    best = max(counts.values())
    return next(a for a in answers if counts[a] == best)        # ties go to the answer seen first
