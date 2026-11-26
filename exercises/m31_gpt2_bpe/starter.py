from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def load_gpt2_ranks():
    """GPT-2's 50,000 learned merges, in the order they were learned. Nothing to change.

    Returns {(left, right): rank}. Rank 0 was learned first. In this file a leading space
    is written as the character 'Ġ', and other bytes are mapped to printable characters.
    """
    path = hf_hub_download("openai-community/gpt2", "merges.txt")
    lines = open(path, encoding="utf-8").read().split("\n")[1:]
    return {tuple(line.split()): rank for rank, line in enumerate(lines) if line}


def bpe_word(word, ranks):
    """Apply the merges to ONE pre-token, e.g. 'Ġstrawberry'. Returns a list of strings.

    Start from the single characters. While some adjacent pair is in `ranks`, take the pair
    with the LOWEST rank and merge every occurrence of it.
    """
    symbols = list(word)
    # TODO 1
    raise NotImplementedError


def gpt2_tokenize(text, tok, ranks):
    """Tokenize a whole text like GPT-2 does. Returns a list of token strings.

    Step 1 (given by the library): split the text into pre-tokens with
        tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)   -> [(pre_token, span), ...]
    Step 2 (yours): run bpe_word on every pre-token and concatenate the results.
    """
    # TODO 2
    raise NotImplementedError
