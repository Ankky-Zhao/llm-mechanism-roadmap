import math
import torch
from transformers import AutoModel, AutoTokenizer

# 1. BM25 on a corpus small enough to check by hand
docs = ["the cat sat", "the dog sat on the mat", "cats and dogs"]
bm = BM25(docs)
assert bm.N == 3 and abs(bm.avgdl - 4.0) < 1e-9
assert abs(bm.idf["cat"] - math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)) < 1e-9          # in 1 of 3 documents
assert abs(bm.idf["the"] - math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)) < 1e-9          # in 2 of 3 documents
assert bm.idf["cat"] > bm.idf["the"], "a rarer term carries more weight"
s = bm.scores("the cat")
norm0 = 1.5 * (1 - 0.75 + 0.75 * 3 / 4)
expected0 = bm.idf["the"] * 1 * 2.5 / (1 + norm0) + bm.idf["cat"] * 1 * 2.5 / (1 + norm0)
norm1 = 1.5 * (1 - 0.75 + 0.75 * 6 / 4)
expected1 = bm.idf["the"] * 2 * 2.5 / (2 + norm1)
assert abs(s[0] - expected0) < 1e-9 and abs(s[1] - expected1) < 1e-9 and s[2] == 0
assert s[0] > s[1] > s[2]
assert bm.scores("zebra") == [0, 0, 0]
# term frequency saturates: ten occurrences are worth less than three times one occurrence
sat = BM25(["x " * 10 + "pad pad", "x pad pad pad pad pad pad pad pad pad pad pad", "pad"])
assert sat.scores("x")[1] < sat.scores("x")[0] < 3 * sat.scores("x")[1]

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

bm = BM25(CORPUS)
rank = lambda scores: sorted(range(len(scores)), key=lambda i: -scores[i])
assert rank(bm.scores("What does error code LH-204 mean?"))[0] == 10, "an exact identifier is what keyword search is good at"

# 2. Embeddings
NAME = "sentence-transformers/all-MiniLM-L6-v2"          # 23 million parameters, Apache 2.0
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME).eval()
D = embed(model, tok, CORPUS)
assert D.shape == (30, 384) and torch.allclose(D.norm(dim=-1), torch.ones(30), atol=1e-5)
alone = embed(model, tok, [CORPUS[17]])[0]
assert torch.allclose(alone, D[17], atol=1e-4), "padding must not change an embedding: use the attention mask when averaging"
v = embed(model, tok, ["Who established the place where people look at the stars?", "a place for watching stars", "the price of bread"])
assert (v[1] @ D[0]) > (v[2] @ D[0]) + 0.2

# 3. The two methods fail on different questions.
q = "Who established the place where people look at the stars?"       # answered by passage 0, with no word in common
r_bm25 = rank(bm.scores(q)).index(0) + 1
r_dense = rank(dense_scores(D, embed(model, tok, [q])[0])).index(0) + 1
assert r_dense == 1 and r_bm25 > 3, (r_dense, r_bm25)
q2 = "What does error code LH-204 mean?"
r2_bm25 = rank(bm.scores(q2)).index(10) + 1
r2_dense = rank(dense_scores(D, embed(model, tok, [q2])[0])).index(10) + 1
assert r2_bm25 == 1

try:                                                       # optional: the same search with FAISS, if it is installed
    import faiss
    index = faiss.IndexFlatIP(384)
    index.add(D.numpy())
    _, ids = index.search(embed(model, tok, [q]).numpy(), 3)
    assert ids[0].tolist() == rank(dense_scores(D, embed(model, tok, [q])[0]))[:3]
    note = "FAISS returns the same top 3"
except ImportError:
    note = "FAISS not installed, skipped"

print(f"m72_bm25: all tests passed ({note})")
print(f"  {q!r}: gold passage at rank {r_bm25} for BM25, rank {r_dense} for embeddings")
print(f"  {q2!r}: gold passage at rank {r2_bm25} for BM25, rank {r2_dense} for embeddings")
