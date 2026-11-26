import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_block(block, x):
    out = block(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def manual_logits(model, input_ids):
    t = model.transformer
    T = input_ids.shape[1]
    x = t.wte(input_ids) + t.wpe(torch.arange(T))     # one 768-d vector per token
    for block in t.h:                                 # 12 rounds of attention + MLP
        x = run_block(block, x)
    return model.lm_head(t.ln_f(x))                   # one score per vocabulary entry


@torch.no_grad()
def next_token_top_k(model, tokenizer, text, k=5):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    last = manual_logits(model, input_ids)[0, -1]     # only the last position predicts the next token
    probs = torch.softmax(last, dim=-1)
    top = probs.topk(k)
    return [(tokenizer.decode(i), p.item()) for p, i in zip(top.values, top.indices)]
