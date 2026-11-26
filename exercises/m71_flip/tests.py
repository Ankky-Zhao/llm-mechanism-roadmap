import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assert relabel([("a", "positive"), ("b", "negative")], {"positive": "X", "negative": "Y"}) == [("a", "X"), ("b", "Y")]

NAME = "Qwen/Qwen2.5-0.5B-Instruct"          # 0.5 billion parameters, Apache 2.0; a download of about 1 GB
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME).eval()

POS = ["The food was wonderful and the staff were friendly.", "I loved every minute of this film.", "An excellent book, beautifully written.",
       "Great value and fast delivery, highly recommended.", "The hotel room was clean, quiet and comfortable.", "This is the best coffee I have had in years.",
       "A fantastic concert with a brilliant band.", "The new phone is fast and the camera is superb.", "Our guide was helpful and the tour was fascinating.",
       "The cake was delicious and looked amazing.", "Wonderful service, we will definitely come back.", "A charming story with a perfect ending."]
NEG = ["Terrible service, I will never come back.", "The battery died after two days. Very disappointing.", "The room was dirty and the bed was broken.",
       "Boring, slow and far too long.", "The soup was cold and tasted awful.", "A waste of money, it broke within a week.", "The staff were rude and unhelpful.",
       "The plot made no sense and the acting was bad.", "My order arrived late and damaged.", "The app crashes constantly and is useless.",
       "Overpriced, noisy and uncomfortable.", "The worst haircut I have ever had."]
TEST = [(t, "positive") for t in POS[4:]] + [(t, "negative") for t in NEG[4:]]
P, N = [(t, "positive") for t in POS[:4]], [(t, "negative") for t in NEG[:4]]
DEMOS4 = [P[0], N[0], N[1], P[1]]
DEMOS8 = [P[0], N[0], N[1], P[1], P[2], N[2], N[3], P[3]]

assert classify(model, tok, build_prompt(DEMOS4, "I loved it."), ["positive", "negative"]) == "positive"

MAPPINGS = {"true labels": {"positive": "positive", "negative": "negative"},
            "flipped labels": {"positive": "negative", "negative": "positive"},
            "neutral labels A / B": {"positive": "A", "negative": "B"},
            "nonsense labels foo / bar": {"positive": "foo", "negative": "bar"}}
rate = {(name, k): follow_rate(model, tok, demos, TEST, mapping)
        for name, mapping in MAPPINGS.items() for k, demos in ((4, DEMOS4), (8, DEMOS8))}

assert rate[("true labels", 4)] >= 0.9
assert rate[("flipped labels", 4)] <= 0.25, "with four flipped examples this model should still answer from its own knowledge"
assert rate[("flipped labels", 8)] > rate[("flipped labels", 4)], "more flipped examples should pull it towards the examples"
assert rate[("neutral labels A / B", 8)] >= 0.8, "labels without a meaning of their own can only be learned from the examples"

print("m71_flip: all tests passed")
print("  share of 16 test reviews answered according to the mapping shown in the examples")
print("                                 4 examples   8 examples")
for name in MAPPINGS:
    print(f"  {name:28s} {rate[(name, 4)]:8.2f}     {rate[(name, 8)]:8.2f}")
