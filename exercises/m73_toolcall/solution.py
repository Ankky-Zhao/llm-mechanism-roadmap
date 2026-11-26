import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_tool_calls(text):
    calls = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S):
        try:
            obj = json.loads(block)
            calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
        except (json.JSONDecodeError, KeyError, TypeError):
            calls.append({"name": None, "arguments": None, "error": "could not parse: " + block})
    return calls


@torch.no_grad()
def generate_reply(model, tokenizer, messages, tools=None, max_new_tokens=120):
    enc = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def tool_round(model, tokenizer, registry, question):
    messages = [{"role": "user", "content": question}]
    first = generate_reply(model, tokenizer, messages, registry.schemas())
    calls = parse_tool_calls(first)
    if not calls:
        return first, [], first
    messages.append({"role": "assistant", "content": "",
                     "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
    for c in calls:
        messages.append({"role": "tool", "name": c["name"], "content": registry.call(c["name"], c["arguments"])})
    final = generate_reply(model, tokenizer, messages, registry.schemas())
    return first, calls, final
