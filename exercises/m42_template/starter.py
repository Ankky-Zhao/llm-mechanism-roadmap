from transformers import AutoTokenizer


def chatml(messages, add_generation_prompt=False):
    """Format a conversation in ChatML, the template used by many open chat models.

    messages: list of {"role": "system" | "user" | "assistant", "content": str}
    Every message becomes   <|im_start|>ROLE\\nCONTENT<|im_end|>\\n
    With add_generation_prompt the string ends with   <|im_start|>assistant\\n
    so that the model continues by writing the assistant's reply.
    """
    # TODO 1
    raise NotImplementedError


def assistant_mask(tok, messages):
    """Which tokens should be trained on?

    Tokenize chatml(messages) with tok(text, add_special_tokens=False). Return (input_ids, mask), two
    lists of equal length. mask[i] is True for the tokens of every assistant message's CONTENT and
    its closing <|im_end|>, and False everywhere else (system and user turns, role headers, newlines).

    Hint: build the text piece by piece and tokenize each piece separately, so you know which
    pieces belong to the assistant.
    """
    # TODO 2
    raise NotImplementedError
