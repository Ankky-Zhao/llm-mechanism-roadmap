import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
PHONE = re.compile(r"\+\d{1,3}[ -]?\(?\d{1,4}\)?(?:[ -]?\d{2,4}){2,4}")


def luhn_valid(number):
    digits = [int(ch) for ch in number if ch.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return len(digits) >= 13 and total % 10 == 0


def redact(text):
    text = EMAIL.sub("[EMAIL]", text)
    text = IBAN.sub("[IBAN]", text)
    text = CARD.sub(lambda m: "[CARD]" if luhn_valid(m.group()) else m.group(), text)
    text = PHONE.sub("[PHONE]", text)
    return text


@torch.no_grad()
def pronoun_gap(model, tokenizer, occupation, template="The {} said that"):
    ids = tokenizer(template.format(occupation), return_tensors="pt").input_ids
    logp = torch.log_softmax(model(ids).logits[0, -1], dim=-1)
    return (logp[tokenizer(" he").input_ids[0]] - logp[tokenizer(" she").input_ids[0]]).item()
