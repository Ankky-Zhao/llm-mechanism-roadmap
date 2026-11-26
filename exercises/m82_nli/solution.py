import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@torch.no_grad()
def entailment_prob(model, tokenizer, premise, hypothesis):
    logits = model(**tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)).logits[0]
    probs = torch.softmax(logits, dim=-1)
    index = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]
    return probs[index].item()


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def groundedness(model, tokenizer, answer, passages):
    scores = []
    for sentence in split_sentences(answer):
        scores.append(max(entailment_prob(model, tokenizer, p, sentence) for p in passages))
    return min(scores), scores
