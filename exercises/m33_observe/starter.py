import tiktoken


def token_bytes(enc, text):
    """The raw bytes of every token of `text`: a list of bytes objects.

    Use enc.encode(text) and enc.decode_single_token_bytes(token_id).
    """
    # TODO 1
    raise NotImplementedError


def tokens_per_char(enc, text):
    """Average number of tokens per character of `text` (a float)."""
    # TODO 2
    raise NotImplementedError


def digit_groups(enc, number):
    """How a string of digits is cut up: the number of digits in each token, as a list of ints."""
    # TODO 3
    raise NotImplementedError


def spell_out(word):
    """'cat' -> ' c a t': a space before every letter."""
    # TODO 4
    raise NotImplementedError


def letters_are_tokens(enc, word):
    """True if, after spell_out, every letter of `word` is a token of its own."""
    # TODO 5
    raise NotImplementedError
