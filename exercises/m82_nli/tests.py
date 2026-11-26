import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

assert split_sentences("One. Two!  Three? ") == ["One.", "Two!", "Three?"] and split_sentences("No stop") == ["No stop"]

NAME = "cross-encoder/nli-deberta-v3-small"          # 142 million parameters, Apache 2.0
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME).eval()

P_ARCHIVE = "The sediment archive of the institute holds 14,300 core samples stored at four degrees Celsius."
P_MARKET = "The weekly market is held on Wednesdays and Saturdays from 7 am to 1 pm on Market Square."
P_FORM = "The sample submission form of the Varnholt Institute is called form S-17 and must be sent two weeks in advance."
P_BRIDGE = "The Sorrel Bridge, built from sandstone in 1754, is the oldest crossing over the river Alme."
P_BUS = "Bus line 47 connects Lindenhof station with the observatory every twenty minutes."
P_HILL = "The Tessin hills reach their highest point at the Rabenkopf, 847 metres above sea level."
# (passage, sentence, is the sentence supported by the passage?)
CASES = [(P_ARCHIVE, "The core samples are kept at four degrees Celsius.", True),
         (P_ARCHIVE, "The drilling mud cylinders are kept at 14°C.", False),                      # a wrong answer from the mini-RAG of 7.2
         (P_MARKET, "The market takes place on Wednesdays and Saturdays.", True),
         (P_MARKET, "On Saturdays and Sundays, you can buy fresh vegetables from stalls in the town centre.", False),   # from 7.2
         (P_FORM, "Form S-17 is used to submit samples to the Varnholt Institute.", True),
         (P_FORM, "Form S-17 is used to report damaged sample containers to the archive staff.", False),               # from 7.2
         (P_BRIDGE, "The Sorrel Bridge was built in 1754.", True),
         (P_BRIDGE, "The Sorrel Bridge was built in 1974.", False),
         (P_BUS, "Bus line 47 runs between the station and the observatory.", True),
         (P_BUS, "Bus line 47 goes from the observatory to the Brunnen stadium.", False),
         (P_HILL, "The Rabenkopf is 847 metres high.", True),
         (P_HILL, "The Rabenkopf is the highest mountain in Europe.", False)]

p = entailment_prob(model, tok, P_BRIDGE, "The Sorrel Bridge was built in 1754.")
assert isinstance(p, float) and p > 0.9
assert entailment_prob(model, tok, P_BRIDGE, "The Sorrel Bridge was built in 1974.") < 0.1, "one digit apart, and contradicted"
probs = [entailment_prob(model, tok, premise, hyp) for premise, hyp, _ in CASES]
correct = sum((pr > 0.5) == gold for pr, (_, _, gold) in zip(probs, CASES))
assert correct >= 11, correct

# An answer of two sentences, one supported and one not
score, per = groundedness(model, tok, "The Sorrel Bridge was built in 1754. It was designed by Clara Wendt.", [P_BRIDGE, P_HILL])
assert len(per) == 2 and per[0] > 0.9 and per[1] < 0.5 and score == min(per)
score2, _ = groundedness(model, tok, "The Rabenkopf is 847 metres high. The Sorrel Bridge was built in 1754.", [P_BRIDGE, P_HILL])
assert score2 > 0.9, "each sentence may be supported by a different passage"

print(f"m82_nli: all tests passed ({correct} of {len(CASES)} sentences classified correctly)")
for pr, (_, hyp, gold) in zip(probs, CASES):
    print(f"  entailment {pr:.2f}   supported: {'yes' if gold else 'no '}   {hyp}")
