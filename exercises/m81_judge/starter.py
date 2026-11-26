import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def judge_prompt(question, answer_a, answer_b):
    """The text shown to the judge. Nothing to change here."""
    return (f"Question: {question}\n\nAnswer A: {answer_a}\nAnswer B: {answer_b}\n\n"
            "Which answer is correct? Reply with a single letter, A or B.")


@torch.no_grad()
def judge_margin(model, tokenizer, question, answer_a, answer_b):
    """How strongly the judge prefers A over B: logit of the token "A" minus logit of the token "B"
    at the position where the reply starts. Positive means the judge says A.

    Render judge_prompt as a user message with the chat template (add_generation_prompt=True, tokenize=False),
    tokenise, run the model once and read the logits at the last position. No generation is needed.
    """
    # TODO 1
    raise NotImplementedError


def position_stats(margins):
    """margins: one pair (m1, m2) per question.
         m1 = judge_margin with the GOOD answer shown as A,   m2 = judge_margin with the GOOD answer shown as B.
       A judge without position bias has m1 > 0 and m2 < 0.

    Returns a dict with four fractions:
      "accuracy_good_first":   share of questions with m1 > 0
      "accuracy_good_second":  share of questions with m2 < 0
      "picks_first":           share of all 2n judgements in which the judge says A
      "consistent":            share of questions where the same ANSWER wins in both orders
    """
    # TODO 2
    raise NotImplementedError


def debiased_accuracy(margins):
    """Judge every pair in both orders and combine: the good answer wins if m1 - m2 > 0.
    A constant preference for position A adds the same amount to m1 and m2 and cancels in the difference."""
    # TODO 3
    raise NotImplementedError
