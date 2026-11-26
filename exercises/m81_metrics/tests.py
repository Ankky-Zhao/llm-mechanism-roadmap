import math

# 1. Normalisation, exact match, F1
assert normalize("The  Sorrel Bridge!") == "sorrel bridge" and normalize("An apple, a pear.") == "apple pear"
assert normalize("theatre") == "theatre", "remove articles as whole words only"
assert exact_match("the Sorrel bridge.", "Sorrel Bridge") == 1.0 and exact_match("1754", "1974") == 0.0
assert abs(token_f1("Paris France", "Paris") - 2 / 3) < 1e-9
assert token_f1("green", "red") == 0.0 and token_f1("", "") == 1.0 and token_f1("", "x") == 0.0
assert abs(token_f1("the cat the cat", "cat") - 2 * 0.5 * 1 / 1.5) < 1e-9, "shared tokens count with multiplicity"

# 2. BLEU, checked by hand
c, r = "the cat sat on the mat", "the cat is on the mat"
assert abs(bleu(c, r, max_n=1) - 5 / 6) < 1e-9                       # 5 of 6 unigrams match
assert abs(bleu(c, r, max_n=2) - math.sqrt(5 / 6 * 3 / 5)) < 1e-9     # 3 of 5 bigrams match
assert abs(bleu("the the the the", "the cat", max_n=1) - 0.25) < 1e-9, "clipping: 'the' may count only once"
assert abs(bleu("the cat", "the cat sat on the mat", max_n=1) - math.exp(1 - 6 / 2)) < 1e-9, "brevity penalty"
assert bleu(r, r) == 1.0 and bleu("a dog", "the cat") == 0.0
assert bleu(c, r, max_n=4) == 0.0 or bleu(c, r, max_n=4) < bleu(c, r, max_n=2)

# 3. ROUGE-L
assert lcs_length("the cat sat on the mat".split(), "the cat is on the mat".split()) == 5
assert lcs_length([], ["a"]) == 0 and lcs_length(list("abcbdab"), list("bdcaba")) == 4
assert abs(rouge_l(c, r) - 5 / 6) < 1e-9 and rouge_l("x y", "a b") == 0.0

# 4. What overlap metrics see and what they miss
reference = "The Sorrel Bridge was built in 1754."
wrong = "The Sorrel Bridge was built in 1974."          # one character off, and false
right = "It dates from 1754."                           # correct, in other words
rows = [(name, exact_match(x, reference), token_f1(x, reference), bleu(x, reference, 2), rouge_l(x, reference))
        for name, x in (("wrong year, same wording", wrong), ("right year, other wording", right))]
assert rows[0][2] > 0.8 and rows[0][4] > 0.8 and rows[1][2] < 0.5 and rows[1][4] < 0.5, "overlap rewards the wording, not the fact"
print("m81_metrics: all tests passed")
print("  reference:", reference)
print("                               EM    F1    BLEU-2  ROUGE-L")
for name, em, f1, b, rl in rows:
    print(f"  {name:27s} {em:4.0f}  {f1:5.2f}  {b:6.2f}  {rl:7.2f}")
