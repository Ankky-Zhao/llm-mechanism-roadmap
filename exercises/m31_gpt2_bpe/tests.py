import tiktoken
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
ranks = load_gpt2_ranks()
assert len(ranks) == 50000 and ranks[("Ġ", "t")] == 0, "the first merge GPT-2 learned is space + t"

assert bpe_word("Ġstrawberry", ranks) == ["Ġstrawberry"], "with a leading space this common word is one token"
assert bpe_word("strawberry", ranks) == ["st", "raw", "berry"], "without the space it falls apart into three"
assert bpe_word("1234567", ranks) == ["123", "45", "67"]

texts = ["The capital of France is Paris.",
         "Tokenization isn't obvious: strawberry vs. strawberry!",
         "1234567 + 7654321 = 8888888",
         "北京是中国的首都。München liegt in Bayern.",
         "def f(x):\n    return x ** 2  # square"]
enc = tiktoken.get_encoding("gpt2")
for t in texts:
    mine = gpt2_tokenize(t, tok, ranks)
    assert mine == tok.tokenize(t), f"differs from the Hugging Face tokenizer on {t!r}"
    assert tok.convert_tokens_to_ids(mine) == enc.encode(t), f"token ids differ from tiktoken on {t!r}"
print("m31_gpt2_bpe: all tests passed. Your code reproduces GPT-2's tokenizer.")
print(" ", gpt2_tokenize("Tokenization isn't obvious", tok, ranks))
