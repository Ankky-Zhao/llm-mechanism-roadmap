import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. The statistics on made-up margins
fair = [(2.0, -1.5), (0.5, -0.5), (3.0, -2.0), (1.0, -1.0)]
assert position_stats(fair) == {"accuracy_good_first": 1.0, "accuracy_good_second": 1.0, "picks_first": 0.5, "consistent": 1.0}
biased = [(4.0, 1.0), (3.0, 2.5), (5.0, 0.5), (2.0, 3.0)]            # always says A; the last pair is truly misjudged
s = position_stats(biased)
assert s == {"accuracy_good_first": 1.0, "accuracy_good_second": 0.0, "picks_first": 1.0, "consistent": 0.0}
assert debiased_accuracy(biased) == 0.75 and debiased_accuracy(fair) == 1.0

# 2. A small model as a judge. In every pair one answer is plainly right and one plainly wrong.
NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME).eval()
PAIRS = [("What is the capital of France?", "Paris is the capital of France.", "The capital of France is Lyon."),
         ("How many days are there in a week?", "A week has seven days.", "A week has ten days."),
         ("What do bees make?", "Bees make honey.", "Bees make milk."),
         ("Which planet is closest to the Sun?", "Mercury is the closest planet to the Sun.", "Jupiter is the closest planet to the Sun."),
         ("What is 12 plus 9?", "12 plus 9 is 21.", "12 plus 9 is 25."),
         ("What colour do you get by mixing blue and yellow?", "Mixing blue and yellow gives green.", "Mixing blue and yellow gives red."),
         ("In which season does snow usually fall?", "Snow usually falls in winter.", "Snow usually falls in summer."),
         ("What is the boiling point of water at sea level in degrees Celsius?", "Water boils at 100 degrees Celsius at sea level.", "Water boils at 40 degrees Celsius at sea level."),
         ("Which animal is known as man's best friend?", "The dog is known as man's best friend.", "The shark is known as man's best friend."),
         ("How many legs does a spider have?", "A spider has eight legs.", "A spider has five legs."),
         ("What is the opposite of hot?", "The opposite of hot is cold.", "The opposite of hot is loud."),
         ("Which ocean lies between Europe and America?", "The Atlantic Ocean lies between Europe and America.", "The Indian Ocean lies between Europe and America.")]
m = judge_margin(model, tok, *PAIRS[0])
assert isinstance(m, float)
margins = [(judge_margin(model, tok, q, good, bad), judge_margin(model, tok, q, bad, good)) for q, good, bad in PAIRS]
s = position_stats(margins)
d = debiased_accuracy(margins)
assert s["picks_first"] >= 0.8, "this small judge should show a strong preference for the first position"
assert s["accuracy_good_first"] - s["accuracy_good_second"] >= 0.5, "its verdict should depend heavily on the order"
assert d >= s["accuracy_good_second"]

print("m81_judge: all tests passed")
print(f"  good answer shown first:  judged correctly in {s['accuracy_good_first']:.0%} of 12 pairs")
print(f"  good answer shown second: judged correctly in {s['accuracy_good_second']:.0%}")
print(f"  says A in {s['picks_first']:.0%} of 24 judgements; same answer wins in both orders for {s['consistent']:.0%} of pairs")
print(f"  combining both orders (m1 - m2 > 0): correct in {d:.0%}")
