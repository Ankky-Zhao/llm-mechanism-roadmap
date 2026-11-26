import tiktoken


def count_tokens(enc, text):
    """Number of tokens of `text` under the tiktoken encoding `enc`."""
    # TODO 1
    raise NotImplementedError


def chat_billed_tokens(user_lens, assistant_lens, system_len=0):
    """Tokens billed over a whole multi-turn conversation.

    Turn i sends the system prompt, every earlier user and assistant message, and the new
    user message as INPUT, and receives assistant_lens[i] tokens as OUTPUT.
    user_lens and assistant_lens are lists of equal length (token counts per turn).
    Returns (total_input_tokens, total_output_tokens).
    """
    # TODO 2
    raise NotImplementedError


def cost(input_tokens, output_tokens, price_in, price_out):
    """Cost in the currency of the prices, which are given per ONE MILLION tokens."""
    # TODO 3
    raise NotImplementedError


def fit_to_window(message_lens, limit, keep_first=True):
    """Drop the oldest messages until the conversation fits into `limit` tokens.

    message_lens: token counts of the messages, oldest first. If keep_first is True the
    first message (the system prompt) is never dropped.
    Returns the list of indices of the messages that are kept, in order.
    """
    # TODO 4
    raise NotImplementedError
