import tiktoken


def count_tokens(enc, text):
    return len(enc.encode(text))


def chat_billed_tokens(user_lens, assistant_lens, system_len=0):
    total_in = total_out = 0
    history = system_len
    for u, a in zip(user_lens, assistant_lens):
        history += u                 # the new user message joins the history
        total_in += history          # the whole history is sent again
        total_out += a
        history += a                 # the answer becomes history for the next turn
    return total_in, total_out


def cost(input_tokens, output_tokens, price_in, price_out):
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def fit_to_window(message_lens, limit, keep_first=True):
    kept = list(range(len(message_lens)))
    start = 1 if keep_first else 0
    while sum(message_lens[i] for i in kept) > limit and len(kept) > start:
        kept.pop(start)              # the oldest message that may be dropped
    return kept
