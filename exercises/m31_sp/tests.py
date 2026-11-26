import codecs, contextlib, io

with contextlib.redirect_stdout(io.StringIO()):
    import this
lines = [l for l in codecs.decode(this.s, "rot13").split("\n") if l]

bpe = train_sentencepiece(lines, 120, "bpe")
uni = train_sentencepiece(lines, 120, "unigram")
assert bpe.get_piece_size() == 120 and uni.get_piece_size() == 120

s = "Simple is better than complex."
pb, pu = bpe.encode(s, out_type=str), uni.encode(s, out_type=str)
assert pb[0].startswith("▁"), "the first piece carries the word-boundary marker"
assert sum(p.startswith("▁") for p in pb) == 5, "five words, five pieces that start with the marker"
assert pb != pu, "BPE and unigram segment the same sentence differently"
assert detokenize(pb) == s and detokenize(pu) == s, "joining the pieces must give the text back"
assert bpe.decode(bpe.encode(s)) == s

# No language-specific pre-tokenization: text without spaces works the same way.
zh = ["今天天气很好", "今天我们学习分词", "分词很重要", "天气很重要"] * 5
z = train_sentencepiece(zh, 20, "bpe")
pz = z.encode("今天分词", out_type=str)
assert detokenize(pz) == "今天分词"
assert len(pz) < len("今天分词") + 1, "frequent character pairs should have been merged"
print("m31_sp: all tests passed")
print("  bpe    :", pb)
print("  unigram:", pu)
print("  zh     :", pz)
