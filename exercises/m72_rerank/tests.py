import time
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

CORPUS = [
    "The Lindenhof Observatory was founded in 1887 by the astronomer Clara Wendt.",
    "The main telescope of the Lindenhof Observatory is a 62-centimetre refractor named Greta.",
    "Visitors can tour the observatory on Thursday evenings; tickets cost 9 euros.",
    "The Varnholt Institute studies river sediments and is located on Mill Street in Lindenhof.",
    "Dr. Jonas Erken has directed the Varnholt Institute since 2019.",
    "The sediment archive of the institute holds 14,300 core samples stored at four degrees Celsius.",
    "Bus line 47 connects Lindenhof station with the observatory every twenty minutes.",
    "The Lindenhof town library closes at 6 pm on weekdays and stays closed on Sundays.",
    "The annual Lantern Festival in Lindenhof takes place on the second Saturday of October.",
    "The Sorrel Bridge, built from sandstone in 1754, is the oldest crossing over the river Alme.",
    "Error code LH-204 on the Greta telescope control unit means that the dome shutter failed to open.",
    "Error code LH-311 indicates that the tracking motor has overheated and must cool for thirty minutes.",
    "The bakery Kornblume on Market Square is known for its plum cake, sold only in September.",
    "The drinking water of Lindenhof comes from the Hallern reservoir, eleven kilometres north of the town.",
    "The sample submission form of the Varnholt Institute is called form S-17 and must be sent two weeks in advance.",
    "The Alme river floods most often in March, when snow melts in the Tessin hills.",
    "The football club of the town, SV Lindenhof 1921, plays its home matches at the Brunnen stadium.",
    "The Brunnen stadium has a capacity of 3,800 spectators.",
    "Mayor Ruth Albrecht was elected in 2022 for a term of six years.",
    "The weekly market is held on Wednesdays and Saturdays from 7 am to 1 pm on Market Square.",
    "The library of the observatory keeps the notebooks of Clara Wendt, which record 412 nights of observation.",
    "Parking near the observatory is limited to 30 spaces; the lot closes at midnight.",
    "The Tessin hills reach their highest point at the Rabenkopf, 847 metres above sea level.",
    "The Lindenhof museum of local history is housed in a former salt warehouse on Quay Lane.",
    "Error code LH-207 means that the focus motor has reached its end stop.",
    "Error code LH-240 means that the weather station reports wind above the safe limit.",
    "Error code LH-312 indicates that the tracking encoder has lost its reference position.",
    "Form S-71 of the Varnholt Institute is the request for access to the cold storage room.",
    "Form S-12 is used to report damaged sample containers to the archive staff.",
    "Bus line 74 runs from Lindenhof station to the Brunnen stadium on match days only.",
]
# (question, index of the passage that answers it, a string the right answer must contain)
QUERIES = [
    ("What does error code LH-204 mean?", 10, "shutter"),
    ("What is form S-17 used for?", 14, "submission"),
    ("Since which year has Jonas Erken directed the institute?", 4, "2019"),
    ("What is the capacity of the Brunnen stadium?", 17, "3,800"),
    ("How high is the Rabenkopf?", 22, "847"),
    ("What does LH-311 indicate?", 11, "overheat"),
    ("When was the Sorrel Bridge built?", 9, "1754"),
    ("Where does bus line 47 go?", 6, "observatory"),
    ("What does LH-207 mean?", 24, "focus"),
    ("What does error LH-312 indicate?", 26, "encoder"),
    ("What is form S-71 for?", 27, "cold storage"),
    ("Where does bus line 74 go?", 29, "stadium"),
    ("Who established the place where people look at the stars?", 0, "Wendt"),
    ("How much do I pay for an evening visit to see the telescope?", 2, "9 euros"),
    ("At what temperature are the drilled mud cylinders kept?", 5, "four"),
    ("In which month is the river most likely to overflow its banks?", 15, "March"),
    ("Where does the tap water of the town originate?", 13, "Hallern"),
    ("On which days can I buy fresh vegetables from stalls in the town centre?", 19, "Wednesday"),
    ("Who leads the local government?", 18, "Albrecht"),
    ("When is the yearly celebration with lights held?", 8, "October"),
]

CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"                   # 23 million parameters, Apache 2.0
ce_tok = AutoTokenizer.from_pretrained(CE)
ce = AutoModelForSequenceClassification.from_pretrained(CE).eval()

s = cross_scores(ce, ce_tok, "How high is the Rabenkopf?", [CORPUS[22], CORPUS[12], CORPUS[17]])
assert len(s) == 3 and s[0] > s[1] + 3 and s[0] > s[2] + 3, "the passage that answers the question should score far above the others"
out = rerank(ce, ce_tok, "How high is the Rabenkopf?", CORPUS, [12, 17, 22, 5, 3], top_n=3)
assert out == [22] + [x for x in out[1:3]] + [5, 3] and set(out[:3]) == {12, 17, 22}, "only the first top_n are re-ordered"

NAME = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME).eval()
bm, D = BM25(CORPUS), embed(model, tok, CORPUS)
gold = [g for _, g, _ in QUERIES]
dense, fused, dense_rr, fused_rr = [], [], [], []
t_first = t_rerank = 0.0
for q, _, _ in QUERIES:
    t = time.perf_counter()
    rd = ranking(dense_scores(D, embed(model, tok, [q])[0]))
    t_first += time.perf_counter() - t
    rf = rrf([ranking(bm.scores(q)), rd])
    t = time.perf_counter()
    dense_rr.append(rerank(ce, ce_tok, q, CORPUS, rd))
    t_rerank += time.perf_counter() - t
    dense.append(rd); fused.append(rf); fused_rr.append(rerank(ce, ce_tok, q, CORPUS, rf))

assert mrr(dense_rr, gold) >= mrr(dense, gold) and mrr(fused_rr, gold) > mrr(fused, gold) + 0.05
assert recall_at_k(dense_rr, gold, 1) >= 0.9

print("m72_rerank: all tests passed")
print("                               Recall@1   Recall@3    MRR")
for name, r in (("embeddings", dense), ("embeddings + reranking", dense_rr), ("RRF", fused), ("RRF + reranking", fused_rr)):
    print(f"  {name:26s} {recall_at_k(r, gold, 1):8.2f} {recall_at_k(r, gold, 3):10.2f} {mrr(r, gold):7.2f}")
print(f"  time for 20 questions: first search {t_first:.2f}s (passage vectors precomputed), reranking 8 candidates each {t_rerank:.2f}s")
