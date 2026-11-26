import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assert extract_answer("23 - 9 = 14, then 14 + 14 = 28.\nAnswer: 28") == 28
assert extract_answer("Answer: **1,215** seats") == 1215
assert extract_answer("First Answer: 3. After checking again.\nAnswer: 7") == 7
assert extract_answer("She has 14 euros left.") == 14 and extract_answer("Answer: -2.5") == -2.5
assert extract_answer("I cannot tell.") is None

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME).eval()

PROBLEMS = [("A shop has 23 apples. It sells 9 and then receives 14 more. How many apples does it have?", 28),
            ("Tom has 15 marbles. He gives 6 to Anna and then buys 12 more. How many marbles does Tom have?", 21),
            ("A bus carries 40 passengers. At the first stop 13 get off and 8 get on. How many passengers are on the bus?", 35),
            ("Lena reads 12 pages on Monday and twice as many on Tuesday. How many pages does she read in total?", 36),
            ("A farmer has 48 sheep. He sells half of them and then buys 10. How many sheep does he have?", 34),
            ("There are 5 shelves with 9 books each. 7 books are borrowed. How many books remain?", 38),
            ("Mia has 50 euros. She buys 3 tickets that cost 12 euros each. How much money does she have left?", 14),
            ("A train has 8 cars with 30 seats each. 25 seats are empty. How many seats are taken?", 215)]

first = ask(model, tok, PROBLEMS[0][0], "direct")
assert first.startswith("Answer:") and len(tok(first).input_ids) <= 12, "the direct reply is prefilled with 'Answer:' and kept short"

rows, direct, cot = [], 0, 0
for q, gold in PROBLEMS:
    d = extract_answer(ask(model, tok, q, "direct"))
    c = extract_answer(ask(model, tok, q, "cot"))
    direct += d == gold
    cot += c == gold
    rows.append((gold, d, c))
assert cot >= direct + 2, f"step-by-step reasoning should help this small model clearly: direct {direct}, cot {cot}"
assert cot >= 6

print(f"m71_cot: all tests passed (direct answer {direct} of 8 correct, step by step {cot} of 8 correct)")
print("  correct   direct   step by step")
for gold, d, c in rows:
    print(f"  {gold:7}   {d!s:>6}   {c!s:>6}")
