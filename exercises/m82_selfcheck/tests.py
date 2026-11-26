import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assert overlap("Paris", "paris.") == 1.0 and overlap("Paris", "Lyon") == 0.0 and overlap("", "") == 1.0 and overlap("", "x") == 0.0
assert abs(overlap("Paris France", "Paris") - 2 / 3) < 1e-9
assert consistency(["1754", "1754", "1754"]) == 1.0 and consistency(["1963", "1457", "1969"]) == 0.0
assert abs(consistency(["a b", "a b", "c"]) - 1 / 3) < 1e-9

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME).eval()

xs = sample_answers(model, tok, "What is the capital of France?")
assert len(xs) == 5 and all(isinstance(x, str) for x in xs)
assert sample_answers(model, tok, "What is the capital of France?") == xs, "the same seed must give the same samples"

KNOWN = ["What is the capital of France?", "How many days are there in a week?", "What do bees make?", "Which planet is closest to the Sun?",
         "What colour do you get by mixing blue and yellow?", "How many legs does a spider have?", "What is the opposite of hot?",
         "What is the largest ocean on Earth?"]
INVENTED = ["What is the capacity of the Brunnen stadium in Lindenhof?", "Who founded the Lindenhof Observatory?",
            "How high is the Rabenkopf in the Tessin hills?", "In which year was the Sorrel Bridge over the river Alme built?",
            "Who has directed the Varnholt Institute since 2019?", "Where does the drinking water of Lindenhof come from?",
            "What does error code LH-204 on the Greta telescope mean?", "On which street is the Lindenhof museum of local history?"]
rows = {name: [(q, sample_answers(model, tok, q)) for q in qs] for name, qs in (("well-known facts", KNOWN), ("an invented town", INVENTED))}
score = {name: [consistency(a) for _, a in r] for name, r in rows.items()}
mean = {name: sum(v) / len(v) for name, v in score.items()}
assert mean["well-known facts"] > mean["an invented town"] + 0.1, mean

# How well does a threshold separate the two groups? Count the pairs (known, invented) ranked the right way round.
pairs = [(k, i) for k in score["well-known facts"] for i in score["an invented town"]]
auc = sum((k > i) + 0.5 * (k == i) for k, i in pairs) / len(pairs)

print(f"m82_selfcheck: all tests passed (a known question is more consistent than an invented one in {auc:.0%} of pairs)")
for name in rows:
    print(f"  {name}: mean consistency {mean[name]:.2f}")
    for (q, a), s in list(zip(rows[name], score[name]))[:4]:
        print(f"    {s:.2f}  {q[:44]:44s} {a[:3]}")
