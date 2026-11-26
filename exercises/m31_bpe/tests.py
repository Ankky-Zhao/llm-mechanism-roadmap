import codecs, contextlib, io

assert pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
assert list(pair_counts([5, 6, 5, 6, 7])) == [(5, 6), (6, 5), (6, 7)], "keys in order of first appearance"
assert merge([5, 6, 6, 7, 9, 1], (6, 7), 99) == [5, 6, 99, 9, 1]
assert merge([1, 1, 1, 1, 1], (1, 1), 9) == [9, 9, 1], "non-overlapping, left to right"

# The classic example: aaabdaaabac
m = train_bpe("aaabdaaabac", 259)
assert m == {(97, 97): 256, (256, 97): 257, (257, 98): 258}, f"got {m}"
assert encode("aaabdaaabac", m) == [258, 100, 258, 97, 99], "11 bytes become 5 tokens"
assert decode([258, 100, 258, 97, 99], m) == "aaabdaaabac"
assert len(train_bpe("ab", 300)) == 1, "stop when no pair is left"

with contextlib.redirect_stdout(io.StringIO()):
    import this
text = codecs.decode(this.s, "rot13")                 # the Zen of Python, 856 bytes
lengths = {}
for V in (256, 300, 400, 600):
    merges = train_bpe(text, V)
    assert len(merges) == V - 256
    ids = encode(text, merges)
    assert decode(ids, merges) == text, "encode then decode must give the text back"
    lengths[V] = len(ids)
assert lengths == {256: 856, 300: 499, 400: 292, 600: 92}, f"got {lengths}"

merges = train_bpe(text, 300)
assert list(merges.items())[:3] == [((115, 32), 256), ((32, 116), 257), ((32, 105), 258)], "first merges: 's ', ' t', ' i'"
# Text the tokenizer has never seen still works, because every byte is in the vocabulary.
new = "better than ugly 北京"
ids = encode(new, merges)
assert ids == [276, 32, 117, 103, 108, 269, 229, 140, 151, 228, 186, 172], f"got {ids}"
assert decode(ids, merges) == new
assert decode([229, 140], merges) == chr(0xFFFD), "half a Chinese character is not valid text"
print("m31_bpe: all tests passed")
print("  tokens for the 856-byte text by vocabulary size:", lengths)
print("  pieces:", [decode([i], merges) for i in encode("Simple is better than complex.", merges)])
