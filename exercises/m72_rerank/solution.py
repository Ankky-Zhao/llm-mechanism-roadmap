import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@torch.no_grad()
def cross_scores(model, tokenizer, query, passages):
    enc = tokenizer([query] * len(passages), passages, padding=True, truncation=True, return_tensors="pt")
    return model(**enc).logits[:, 0].tolist()


def rerank(model, tokenizer, query, corpus, candidates, top_n=8):
    head, tail = candidates[:top_n], candidates[top_n:]
    scores = cross_scores(model, tokenizer, query, [corpus[i] for i in head])
    head = [doc for _, doc in sorted(zip(scores, head), key=lambda pair: -pair[0])]
    return head + tail
