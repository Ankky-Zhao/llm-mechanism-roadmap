import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text):
    marked = re.findall(r"Answer:\s*\**\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    numbers = marked or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return None
    return float(numbers[-1].replace(",", "").rstrip("."))


@torch.no_grad()
def ask(model, tokenizer, question, mode, max_new_tokens=180):
    if mode == "direct":
        content, prefill, max_new_tokens = question, "Answer:", 8
    else:
        content = question + " Think step by step. End with a line of the form 'Answer: <number>'."
        prefill = ""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False) + prefill
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return prefill + tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
