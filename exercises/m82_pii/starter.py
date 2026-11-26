import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Part 1: remove personal data from text before it is sent to a model or written to a log.
# Patterns catch data with a fixed shape. They do not catch names or addresses.

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")                       # 13 to 19 digits, optionally grouped
PHONE = re.compile(r"\+\d{1,3}[ -]?\(?\d{1,4}\)?(?:[ -]?\d{2,4}){2,4}")   # international format, starting with +


def luhn_valid(number):
    """The checksum that real card numbers satisfy. Use the digits of `number` only.

    From the right, double every second digit (the last digit is not doubled); if a doubled digit
    exceeds 9, subtract 9. The number is valid if it has at least 13 digits and the sum is divisible by 10.
    """
    # TODO 1
    raise NotImplementedError


def redact(text):
    """Replace, in this order: e-mail addresses by [EMAIL], IBANs by [IBAN], card numbers by [CARD]
    (only if luhn_valid, otherwise leave the digits as they are), phone numbers by [PHONE]."""
    # TODO 2: pattern.sub(replacement, text); for the card, pass a function as replacement
    raise NotImplementedError


# Part 2: a counterfactual probe. Two prompts that differ only in one word are compared.

@torch.no_grad()
def pronoun_gap(model, tokenizer, occupation, template="The {} said that"):
    """log p(" he") - log p(" she") as the next token after template.format(occupation).
    Positive: the model expects "he" more than "she" after this occupation."""
    # TODO 3
    raise NotImplementedError
