import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCHEMA = {"type": "object", "properties": {
    "sentiment": {"enum": ["positive", "negative"]},
    "stars": {"type": "integer", "maxDigits": 1},
    "recommend": {"type": "boolean"}}}

# 1. The checking functions
assert extract_json('Sure! {"a": 1} and more {"b": 2}') == {"a": 1}
assert extract_json("no json here") is None and extract_json('{"a": 1') is None and extract_json("{'a': 1}") is None
ok = {"sentiment": "positive", "stars": 4, "recommend": True}
assert conforms(ok, SCHEMA)
for bad in [None, [], {**ok, "sentiment": "great"}, {**ok, "stars": "4"}, {**ok, "stars": True}, {**ok, "recommend": "yes"},
            {**ok, "extra": 1}, {"sentiment": "positive", "stars": 4}]:
    assert not conforms(bad, SCHEMA), bad
assert format_error_rate([json.dumps(ok), "oops", '{"stars": 4}', "x " + json.dumps(ok) + " y"], SCHEMA) == 0.5
assert field_accuracy([json.dumps(ok), "oops", json.dumps(ok)], "sentiment", ["positive", "positive", "negative"]) == 1 / 3

# 2. The experiment on GPT-2
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
strings = tok.batch_decode([[i] for i in range(len(tok))])
dfa = compile_schema(SCHEMA)

REVIEWS = [("The food was wonderful and the staff were friendly.", "positive"), ("Terrible service, I will never come back.", "negative"),
           ("I loved every minute of this film.", "positive"), ("The battery died after two days. Very disappointing.", "negative"),
           ("An excellent book, beautifully written.", "positive"), ("The room was dirty and the bed was broken.", "negative"),
           ("Great value and fast delivery, highly recommended.", "positive"), ("Boring, slow and far too long.", "negative")]
labels = [label for _, label in REVIEWS]
ZERO = 'Review: {r}\nJSON with the fields sentiment, stars and recommend:'
SHOT = ('Review: The pizza was cold and the waiter was rude.\nJSON: {{"sentiment": "negative", "stars": 1, "recommend": false}}\n\n'
        'Review: A lovely hotel with a fantastic view.\nJSON: {{"sentiment": "positive", "stars": 5, "recommend": true}}\n\n'
        'Review: {r}\nJSON:')

def free(prompt):
    ids = tok(prompt, return_tensors="pt").input_ids
    return tok.decode(model.generate(ids, do_sample=False, max_new_tokens=24, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:])

def constrained(prompt):
    ids = tok(prompt + " ", return_tensors="pt").input_ids
    return constrained_greedy(model, strings, dfa, ids, 40, tok.eos_token_id)

runs = {"zero-shot, unconstrained": [free(ZERO.format(r=r)) for r, _ in REVIEWS],
        "two-shot, unconstrained": [free(SHOT.format(r=r)) for r, _ in REVIEWS],
        "zero-shot, constrained": [constrained(ZERO.format(r=r)) for r, _ in REVIEWS],
        "two-shot, constrained": [constrained(SHOT.format(r=r)) for r, _ in REVIEWS]}
err = {k: format_error_rate(v, SCHEMA) for k, v in runs.items()}
acc = {k: field_accuracy(v, "sentiment", labels) for k, v in runs.items()}

assert err["zero-shot, constrained"] == 0 and err["two-shot, constrained"] == 0, "constrained decoding cannot produce a format error"
assert err["zero-shot, unconstrained"] > 0.5, "without examples or constraints GPT-2 rarely produces the format"
assert all(dfa_walk(dfa, 0, t) in dfa["accept"] for t in runs["zero-shot, constrained"])

print("m53_compare: all tests passed")
print("  setting                      format errors   sentiment correct")
for k in runs:
    print(f"  {k:27s}  {err[k]:8.0%}        {acc[k]:8.0%}")
print("  example, zero-shot unconstrained:", repr(runs["zero-shot, unconstrained"][0]))
print("  example, two-shot unconstrained: ", repr(runs["two-shot, unconstrained"][0]))
print("  example, zero-shot constrained:  ", repr(runs["zero-shot, constrained"][0]))
print("  example, two-shot constrained:   ", repr(runs["two-shot, constrained"][0]))
