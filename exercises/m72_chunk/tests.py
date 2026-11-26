text = " ".join(f"w{i}" for i in range(10))
assert chunk_words(text, 4) == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]
assert chunk_words(text, 4, 2) == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6 w7", "w6 w7 w8 w9"], "no trailing chunk made of overlap only"
assert chunk_words(text, 5, 1) == ["w0 w1 w2 w3 w4", "w4 w5 w6 w7 w8", "w8 w9"]
assert chunk_words(text, 20, 5) == [text] and chunk_words("", 4) == []
for size, ov in ((3, 0), (4, 1), (6, 3), (7, 2)):
    cs = chunk_words(text, size, ov)
    assert all(len(c.split()) <= size for c in cs)
    assert [w for i, c in enumerate(cs) for w in c.split()[(ov if i else 0):]] == text.split(), "every word is covered once outside the overlaps"

s = "One two three. Four five! Six seven eight nine? Ten."
assert chunk_sentences(s, 5) == ["One two three. Four five!", "Six seven eight nine? Ten."]
assert chunk_sentences(s, 4) == ["One two three.", "Four five!", "Six seven eight nine?", "Ten."]
assert chunk_sentences(s, 3) == ["One two three.", "Four five!", "Six seven eight nine?", "Ten."], "a sentence above the limit stands alone"
assert chunk_sentences(s, 100) == [s]

# What overlap is for: the name and the year of one fact sit on either side of a boundary.
doc = ("The Lindenhof observatory stands on the northern hill above the old town and has been open to visitors "
       "for many decades. Its main telescope was built by the instrument maker Clara Voss and it was installed "
       "in 1931 after a long delay caused by a harsh winter. The dome was renovated twice since then.")
both = lambda chunks: [c for c in chunks if "Voss" in c and "1931" in c]
assert len(both(chunk_words(doc, 34, 0))) == 0, "without overlap the boundary separates the two halves"
assert len(both(chunk_words(doc, 34, 8))) >= 1, "with overlap one chunk holds both"
assert len(both(chunk_sentences(doc, 34))) == 1, "cutting at sentence boundaries keeps the sentence whole"
print("ok")
