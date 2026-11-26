import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def judge_prompt(question, answer_a, answer_b):
    return (f"Question: {question}\n\nAnswer A: {answer_a}\nAnswer B: {answer_b}\n\n"
            "Which answer is correct? Reply with a single letter, A or B.")


@torch.no_grad()
def judge_margin(model, tokenizer, question, answer_a, answer_b):
    text = tokenizer.apply_chat_template([{"role": "user", "content": judge_prompt(question, answer_a, answer_b)}],
                                         add_generation_prompt=True, tokenize=False)
    logits = model(tokenizer(text, return_tensors="pt").input_ids).logits[0, -1]
    return (logits[tokenizer("A").input_ids[0]] - logits[tokenizer("B").input_ids[0]]).item()


def position_stats(margins):
    n = len(margins)
    return {"accuracy_good_first": sum(m1 > 0 for m1, _ in margins) / n,
            "accuracy_good_second": sum(m2 < 0 for _, m2 in margins) / n,
            "picks_first": sum((m1 > 0) + (m2 > 0) for m1, m2 in margins) / (2 * n),
            "consistent": sum((m1 > 0) == (m2 < 0) for m1, m2 in margins) / n}


def debiased_accuracy(margins):
    return sum((m1 - m2) > 0 for m1, m2 in margins) / len(margins)
