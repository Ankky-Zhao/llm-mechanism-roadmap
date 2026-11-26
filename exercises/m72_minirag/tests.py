import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

p = build_rag_prompt("Who?", ["First.", "Second."])
assert p == RAG_INSTRUCTION + "\n\n[1] First.\n[2] Second.\n\nQuestion: Who?\nAnswer in one short sentence.", repr(p)

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

embed_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
cross_tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
cross = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2").eval()
LLM = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(LLM)
llm = AutoModelForCausalLM.from_pretrained(LLM).eval()

bm, D = BM25(CORPUS), embed(embedder, embed_tok, CORPUS)
retriever = lambda q: retrieve(q, CORPUS, bm, D, embedder, embed_tok, cross, cross_tok, k=3)
assert retriever("How high is the Rabenkopf?")[0] == 22 and len(retriever("How high is the Rabenkopf?")) == 3

reply = chat(llm, tok, "Reply with the single word: hello")
assert isinstance(reply, str) and "<|im_end|>" not in reply and "hello" in reply.lower()

# The corpus describes an invented town, so the model cannot know any of the answers from training.
correct = lambda reply, needle: needle.lower() in reply.lower()
rows, closed_ok, rag_ok, found = [], 0, 0, 0
for q, gold, needle in QUERIES:
    closed = chat(llm, tok, q + " Answer in one short sentence.")
    reply, ids = rag_answer(q, CORPUS, retriever, llm, tok)
    closed_ok += correct(closed, needle); rag_ok += correct(reply, needle); found += gold in ids
    rows.append((q, gold in ids, correct(reply, needle), closed, reply))
assert closed_ok <= 4, "without the passages the model can only guess"
assert found >= 17 and rag_ok >= 12, (found, rag_ok)

# With passages that do not contain the answer, the instruction asks for "I don't know".
off_topic = chat(llm, tok, build_rag_prompt("What is the capacity of the Brunnen stadium?", [CORPUS[12], CORPUS[7]]))

print(f"m72_minirag: all tests passed")
print(f"  20 questions about an invented town: {closed_ok} answered correctly without retrieval, {rag_ok} with retrieval")
print(f"  the passage with the answer was among the 3 retrieved for {found} questions")
print("  retrieved the right passage and still answered wrongly:")
for q, hit, ok, closed, reply in rows:
    if hit and not ok:
        print(f"    Q: {q}\n       {reply!r}")
print("  an example without retrieval:", repr(rows[3][3]))
print("  the same question with retrieval:", repr(rows[3][4]))
print("  passages without the answer:", repr(off_topic))
