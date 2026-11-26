import tiktoken


def token_bytes(enc, text):
    return [enc.decode_single_token_bytes(t) for t in enc.encode(text)]


def tokens_per_char(enc, text):
    return len(enc.encode(text)) / len(text)


def digit_groups(enc, number):
    return [len(b) for b in token_bytes(enc, number)]


def spell_out(word):
    return "".join(" " + c for c in word)


def letters_are_tokens(enc, word):
    return [b.decode() for b in token_bytes(enc, spell_out(word))] == [" " + c for c in word]
