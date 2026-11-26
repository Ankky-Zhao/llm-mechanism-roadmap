from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")   # a small open chat model that uses ChatML
msgs = [{"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"}]

assert chatml(msgs[:2]) == "<|im_start|>system\nYou are concise.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
for n in (2, 3, 4):
    for gen in (False, True):
        ref = tok.apply_chat_template(msgs[:n], tokenize=False, add_generation_prompt=gen)
        assert chatml(msgs[:n], gen) == ref, f"differs from the model's own template ({n} messages, generation prompt {gen})"

# the control markers are single tokens, not ordinary text
assert len(tok("<|im_start|>", add_special_tokens=False).input_ids) == 1
assert len(tok("<|im_end|>", add_special_tokens=False).input_ids) == 1

conv = msgs + [{"role": "assistant", "content": "4"}]
ids, mask = assistant_mask(tok, conv)
assert len(ids) == len(mask) and ids == tok(chatml(conv), add_special_tokens=False).input_ids, "pieces must add up to the full text"
trained = tok.decode([i for i, m in zip(ids, mask) if m])
assert trained == "Hello!<|im_end|>4<|im_end|>", f"the trained tokens decode to {trained!r}"
assert sum(mask) < len(mask) / 2, "most of this conversation is context, not training target"
print("m42_template: all tests passed")
print(" ", repr(chatml(msgs, True)))
