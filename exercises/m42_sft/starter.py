import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# We fine-tune GPT-2 small, a pure text-continuation model, on a handful of
# question/answer pairs written in a fixed format. Two things to watch:
#   1. before fine-tuning it does not answer, it just continues the text
#   2. after fine-tuning it answers in the format, but models general text a bit worse


def format_prompt(question):
    return f"<|user|>\n{question}\n<|assistant|>\n"


def build_example(tok, question, answer):
    """One training example. Returns (input_ids, labels), two lists of equal length.

    The full text is  format_prompt(question) + f"ANSWER: {answer}." + tok.eos_token
    labels equals input_ids on the answer part (including the end token) and is -100 on the prompt part.
    """
    # TODO 1
    raise NotImplementedError


def collate(examples, pad_id):
    """Pad a list of (input_ids, labels) to one batch. Nothing to change."""
    n = max(len(i) for i, _ in examples)
    ids = torch.full((len(examples), n), pad_id)
    labels = torch.full((len(examples), n), -100)
    attn = torch.zeros((len(examples), n), dtype=torch.long)
    for r, (i, l) in enumerate(examples):
        ids[r, :len(i)] = torch.tensor(i); labels[r, :len(l)] = torch.tensor(l); attn[r, :len(i)] = 1
    return ids, attn, labels


def sft_train(model, tok, pairs, steps=12, lr=1e-4):
    """Supervised fine-tuning on the whole set of pairs as one batch per step.

    Returns the list of loss values. Use torch.optim.AdamW and
    model(input_ids=..., attention_mask=..., labels=...).loss
    """
    # TODO 2
    raise NotImplementedError


@torch.no_grad()
def answer(model, tok, question, max_new_tokens=8):
    """Greedy continuation of the prompt, decoded. Nothing to change."""
    ids = tok(format_prompt(question), return_tensors="pt").input_ids
    out = model.generate(ids, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:])


@torch.no_grad()
def perplexity(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids
    return math.exp(model(ids, labels=ids).loss.item())
