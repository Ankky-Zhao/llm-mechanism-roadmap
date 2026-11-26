import tiktoken

gpt2 = tiktoken.get_encoding("gpt2")              # GPT-2, vocabulary 50,257
o200k = tiktoken.get_encoding("o200k_base")       # a newer OpenAI encoding, vocabulary about 200,000

# 1. Words are not made of letters. The same word is cut differently with and without a space.
assert token_bytes(gpt2, "strawberry") == [b"st", b"raw", b"berry"]
assert token_bytes(gpt2, " strawberry") == [b" strawberry"]
assert b"".join(token_bytes(o200k, "Tokenization isn't obvious")) == b"Tokenization isn't obvious"

# 2. The model can only see letters if the text is written so that each letter is a token.
assert spell_out("cat") == " c a t"
assert letters_are_tokens(gpt2, "strawberry") and letters_are_tokens(o200k, "strawberry")
assert len(gpt2.encode(spell_out("strawberry"))) == 10

# 3. Digits are cut into irregular chunks, and one extra digit shifts the chunks.
assert digit_groups(gpt2, "1234567") == [3, 2, 2]
assert digit_groups(gpt2, "12345678") == [3, 2, 3]
assert digit_groups(o200k, "1234567") == [3, 3, 1], "this encoding cuts from the left in threes"
assert digit_groups(o200k, "12345678901") == [3, 3, 3, 2]

# 4. Languages cost different numbers of tokens, and it depends on the tokenizer.
zh, en = "北京是中国的首都", "Beijing is the capital of China"
assert tokens_per_char(gpt2, zh) > 1.5, "GPT-2 needs almost two tokens per Chinese character"
assert tokens_per_char(o200k, zh) < 1.0
assert tokens_per_char(gpt2, en) < 0.3
assert len(o200k.encode(zh)) < len(o200k.encode(en)), "with the newer encoding this sentence is cheaper in Chinese"

# 5. In GPT-2 a single Chinese character can be split across tokens: some tokens are not valid text on their own.
broken = 0
for b in token_bytes(gpt2, "今天天气很好"):
    try:
        b.decode("utf-8")
    except UnicodeDecodeError:
        broken += 1
assert broken >= 4
print("m33_observe: all tests passed")
print("  gpt2  1234567 ->", [b.decode() for b in token_bytes(gpt2, "1234567")])
print("  o200k 1234567 ->", [b.decode() for b in token_bytes(o200k, "1234567")])
print(f"  tokens per character, zh: gpt2 {tokens_per_char(gpt2, zh):.2f}, o200k {tokens_per_char(o200k, zh):.2f}")
