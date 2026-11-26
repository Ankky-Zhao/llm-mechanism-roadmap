import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Two small automata written by hand.
YES_NO = {"trans": {0: {"y": 1, "n": 4}, 1: {"e": 2}, 2: {"s": 3}, 4: {"o": 3}}, "accept": {3}}          # yes | no
DIGITS = "0123456789"
NUMBER = {"trans": {0: {d: 1 for d in DIGITS}, 1: {d: 2 for d in DIGITS}, 2: {d: 3 for d in DIGITS}}, "accept": {1, 2, 3}}   # 1 to 3 digits

# 1. Walking
assert dfa_walk(YES_NO, 0, "ye") == 2 and dfa_walk(YES_NO, 0, "yes") == 3 and dfa_walk(YES_NO, 2, "s") == 3
assert dfa_walk(YES_NO, 0, "yea") is None and dfa_walk(YES_NO, 3, "s") is None and dfa_walk(YES_NO, 0, "") == 0

# 2. Allowed tokens on a tiny vocabulary: whole words, pieces and characters all work.
vocab = ["yes", "y", "es", "e", "s", "no", "n", "o", "maybe", " yes", "", "ye", "yess"]
names = lambda ids: sorted(vocab[i] for i in ids)
assert names(allowed_tokens(YES_NO, 0, vocab)) == ["n", "no", "y", "ye", "yes"]
assert names(allowed_tokens(YES_NO, 1, vocab)) == ["e", "es"]
assert names(allowed_tokens(YES_NO, 3, vocab)) == []

# 3. On GPT-2
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
strings = tok.batch_decode([[i] for i in range(len(tok))])
n_first = len(allowed_tokens(NUMBER, 0, strings))
assert 500 < n_first < 1110, n_first            # the tokens made of 1 to 3 digits, out of 50257

rows = []
for prompt in ["Question: Is Paris the capital of France? Answer (yes or no):", "Question: Is the Moon made of cheese? Answer (yes or no):"]:
    ids = tok(prompt, return_tensors="pt").input_ids
    free = tok.decode(model.generate(ids, do_sample=False, max_new_tokens=6, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:])
    out = constrained_greedy(model, strings, YES_NO, ids, 6, tok.eos_token_id)
    assert out in ("yes", "no"), out
    rows.append((prompt, free, out))
for prompt in ["The number of days in a week is", "The year the Second World War ended was"]:
    ids = tok(prompt, return_tensors="pt").input_ids
    free = tok.decode(model.generate(ids, do_sample=False, max_new_tokens=6, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:])
    out = constrained_greedy(model, strings, NUMBER, ids, 6, tok.eos_token_id)
    assert re.fullmatch(r"[0-9]{1,3}", out), out
    rows.append((prompt, free, out))

print(f"m53_fsm: all tests passed ({n_first} of {len(strings)} tokens are allowed at the start of a number)")
for prompt, free, out in rows:
    print(f"  {prompt!r}")
    print(f"      unconstrained: {free!r}")
    print(f"      constrained:   {out!r}")
