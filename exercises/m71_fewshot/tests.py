import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. The prompt format, character by character
assert build_prompt([], "Fine.") == "Review: Fine.\nSentiment:"
assert build_prompt([("Good.", "positive"), ("Bad.", "negative")], "Fine.") == \
    "Review: Good.\nSentiment: positive\n\nReview: Bad.\nSentiment: negative\n\nReview: Fine.\nSentiment:"
assert build_prompt([("a", "1")], "b", "Q: {x}\nA:") == "Q: a\nA: 1\n\nQ: b\nA:"

# 2. Voting
assert majority_vote(["28", "27", "28", None, "28"]) == "28"
assert majority_vote(["a", "b", "b", "a"]) == "a" and majority_vote([None, None]) is None and majority_vote([]) is None

# 3. GPT-2 as a classifier
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
LABELS = ["positive", "negative"]
s = label_scores(model, tok, "Review: I loved it.\nSentiment:", LABELS)
with torch.no_grad():
    ref = model(tok("Review: I loved it.\nSentiment:", return_tensors="pt").input_ids).logits[0, -1]
assert len(s) == 2 and abs(s[0] - ref[tok(" positive").input_ids[0]].item()) < 1e-4
assert classify(model, tok, "Review: I loved it.\nSentiment:", LABELS) in LABELS

POS = ["The food was wonderful and the staff were friendly.", "I loved every minute of this film.", "An excellent book, beautifully written.",
       "Great value and fast delivery, highly recommended.", "The hotel room was clean, quiet and comfortable.", "This is the best coffee I have had in years.",
       "A fantastic concert with a brilliant band.", "The new phone is fast and the camera is superb.", "Our guide was helpful and the tour was fascinating.",
       "The cake was delicious and looked amazing.", "Wonderful service, we will definitely come back.", "A charming story with a perfect ending."]
NEG = ["Terrible service, I will never come back.", "The battery died after two days. Very disappointing.", "The room was dirty and the bed was broken.",
       "Boring, slow and far too long.", "The soup was cold and tasted awful.", "A waste of money, it broke within a week.", "The staff were rude and unhelpful.",
       "The plot made no sense and the acting was bad.", "My order arrived late and damaged.", "The app crashes constantly and is useless.",
       "Overpriced, noisy and uncomfortable.", "The worst haircut I have ever had."]
TEST = [(t, "positive") for t in POS[4:]] + [(t, "negative") for t in NEG[4:]]          # 16 reviews, half of each kind
DEMOS = [(POS[0], "positive"), (NEG[0], "negative"), (NEG[1], "negative"), (POS[1], "positive")]

def accuracy(demos, template):
    return sum(classify(model, tok, build_prompt(demos, x, template), LABELS) == y for x, y in TEST) / len(TEST)

# 4. The same task, five ways of asking. Nothing changes except the wording.
TEMPLATES = ["Review: {x}\nSentiment:", "{x}\nThe sentiment of this review is", "Text: {x}\nLabel:",
             "Q: Is this review positive or negative? {x}\nA:", "{x} Overall, it was"]
zero = [accuracy([], t) for t in TEMPLATES]
few = [accuracy(DEMOS, t) for t in TEMPLATES]
assert max(zero) - min(zero) >= 0.25, "the wording alone should move zero-shot accuracy by a wide margin on GPT-2"
assert max(zero) >= 0.85

# 5. Majority-label bias: if all examples carry the same label, the model repeats it.
all_pos = [(t, "positive") for t in POS[:4]]
n_pos = sum(classify(model, tok, build_prompt(all_pos, x), LABELS) == "positive" for x, _ in TEST)
assert n_pos >= 14, n_pos

print("m71_fewshot: all tests passed")
print("  accuracy on 16 reviews (chance = 0.50)             zero-shot   4-shot")
for t, z, f in zip(TEMPLATES, zero, few):
    print(f"  {t!r:52s} {z:6.2f}   {f:6.2f}")
print(f"  with four examples that are all labelled positive, {n_pos} of 16 test reviews are called positive")
