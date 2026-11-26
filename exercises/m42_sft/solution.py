import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def format_prompt(question):
    return f"<|user|>\n{question}\n<|assistant|>\n"


def build_example(tok, question, answer):
    """One training example. Returns (input_ids, labels), two lists of equal length.

    The full text is  format_prompt(question) + f"ANSWER: {answer}." + tok.eos_token
    labels equals input_ids on the answer part (including the end token) and is -100 on the prompt part.
    """
    prompt_ids = tok(format_prompt(question)).input_ids
    answer_ids = tok(f"ANSWER: {answer}." + tok.eos_token).input_ids
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids


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
    ids, attn, labels = collate([build_example(tok, q, a) for q, a in pairs], tok.eos_token_id)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


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
