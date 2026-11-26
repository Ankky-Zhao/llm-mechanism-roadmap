import itertools
import re
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def sample_answers(model, tokenizer, question, n=5, temperature=1.0, seed=0, max_new_tokens=12):
    torch.manual_seed(seed)
    enc = tokenizer.apply_chat_template([{"role": "user", "content": question + " Answer with a few words only."}],
                                        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                         top_p=1.0, top_k=0, num_return_sequences=n)
    return [tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


def overlap(a, b):
    ta = re.sub(r"[^a-z0-9 ]", "", a.lower()).split()
    tb = re.sub(r"[^a-z0-9 ]", "", b.lower()).split()
    if not ta or not tb:
        return float(ta == tb)
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def consistency(answers):
    pairs = list(itertools.combinations(answers, 2))
    return sum(overlap(a, b) for a, b in pairs) / len(pairs)
