from transformers import AutoTokenizer


def chatml(messages, add_generation_prompt=False):
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def assistant_mask(tok, messages):
    ids, mask = [], []
    def add(piece, trainable):
        piece_ids = tok(piece, add_special_tokens=False).input_ids
        ids.extend(piece_ids)
        mask.extend([trainable] * len(piece_ids))
    for m in messages:
        add(f"<|im_start|>{m['role']}\n", False)                        # the header is given, never predicted
        add(f"{m['content']}<|im_end|>", m["role"] == "assistant")      # the reply and its end marker are learned
        add("\n", False)
    return ids, mask
