from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def load_gpt2_ranks():
    path = hf_hub_download("openai-community/gpt2", "merges.txt")
    lines = open(path, encoding="utf-8").read().split("\n")[1:]
    return {tuple(line.split()): rank for rank, line in enumerate(lines) if line}


def bpe_word(word, ranks):
    symbols = list(word)
    while len(symbols) >= 2:
        pairs = set(zip(symbols, symbols[1:]))
        best = min(pairs, key=lambda p: ranks.get(p, float("inf")))
        if best not in ranks:
            break
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


def gpt2_tokenize(text, tok, ranks):
    out = []
    for pre_token, _ in tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text):
        out.extend(bpe_word(pre_token, ranks))
    return out
