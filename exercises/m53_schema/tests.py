import json
import random


def walk(dfa, text):
    s = 0
    for ch in text:
        s = dfa["trans"].get(s, {}).get(ch)
        if s is None:
            return None
    return s

accepts = lambda dfa, text: walk(dfa, text) in dfa["accept"]

def piece(build):
    b = Builder()
    ends = build(b, {0})
    return {"trans": b.trans, "accept": set(ends)}

# 1. The building blocks
d = piece(lambda b, s: literal(b, s, "abc"))
assert accepts(d, "abc") and not accepts(d, "ab") and not accepts(d, "abcd")
d = piece(lambda b, s: choice(b, s, ["true", "false"]))
assert accepts(d, "true") and accepts(d, "false") and not accepts(d, "tru") and not accepts(d, "truefalse")
d = piece(lambda b, s: choice(b, s, ['"red"', '"rose"', '"blue"']))          # shared prefix
assert all(accepts(d, x) for x in ['"red"', '"rose"', '"blue"']) and not accepts(d, '"ro"') and not accepts(d, '"rede"')
d = piece(lambda b, s: integer(b, s, 3))
assert all(accepts(d, x) for x in ["0", "7", "42", "999"])
assert not any(accepts(d, x) for x in ["", "007", "01", "1000", "-1", "4.2"])
d = piece(lambda b, s: free_string(b, s, 5))
assert all(accepts(d, x) for x in ['""', '"a"', '"Hi yo"'])
assert not any(accepts(d, x) for x in ['"', '"abcdef"', '"a"b"', 'abc', '"a\\"'])

# 2. A whole schema
SCHEMA = {"type": "object", "properties": {
    "sentiment": {"enum": ["positive", "negative", "neutral"]},
    "stars": {"type": "integer", "maxDigits": 1},
    "recommend": {"type": "boolean"},
    "summary": {"type": "string", "maxLength": 12}}}
dfa = compile_schema(SCHEMA)
good = '{"sentiment": "negative", "stars": 2, "recommend": false, "summary": "Too slow."}'
assert accepts(dfa, good) and json.loads(good)["stars"] == 2
for bad in [good[:-1], good + "}", good.replace("negative", "bad"), good.replace("false", "False"), good.replace(": 2", ': "2"'),
            good.replace('"stars": 2, ', ""), good.replace("Too slow.", "This summary is far too long"), good.replace('", "stars', '","stars')]:
    assert not accepts(dfa, bad), bad

# 3. Every string the DFA accepts is valid JSON that satisfies the schema. Check by random walks.
def conforms(obj):
    return (isinstance(obj, dict) and list(obj) == ["sentiment", "stars", "recommend", "summary"]
            and obj["sentiment"] in ("positive", "negative", "neutral")
            and type(obj["stars"]) is int and 0 <= obj["stars"] <= 9
            and type(obj["recommend"]) is bool and type(obj["summary"]) is str and len(obj["summary"]) <= 12)

rng = random.Random(0)
seen = set()
for _ in range(300):
    s, text = 0, ""
    while dfa["trans"].get(s):
        if s in dfa["accept"] and rng.random() < 0.5:
            break
        ch = rng.choice(sorted(dfa["trans"][s]))
        text, s = text + ch, dfa["trans"][s][ch]
    assert s in dfa["accept"], "every path must be able to finish"
    assert conforms(json.loads(text)), text
    seen.add(text)
assert len(seen) > 250

# 4. The automaton is finite and has no loops, so generation is guaranteed to end.
longest = {}
def depth(s):
    if s not in longest:
        longest[s] = -1                                        # a loop would hit this marker
        nxt = [depth(t) for t in dfa["trans"].get(s, {}).values()]
        assert -1 not in nxt, "the DFA must not contain a loop"
        longest[s] = 1 + max(nxt) if nxt else 0
    return longest[s]
print(f"m53_schema: all tests passed ({len(dfa['trans'])} states, longest output {depth(0)} characters)")
print("  one random string it accepts:", sorted(seen)[0])
