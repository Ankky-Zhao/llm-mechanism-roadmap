import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Parsing
assert parse_tool_calls("Hello there.") == []
assert parse_tool_calls('<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call>') == [{"name": "multiply", "arguments": {"a": 2, "b": 3}}]
two = parse_tool_calls('Sure.<tool_call>{"name": "a", "arguments": {}}</tool_call> and <tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>')
assert [c["name"] for c in two] == ["a", "b"]
bad = parse_tool_calls('<tool_call>{"name": "multiply", "arguments": {a: 2}}</tool_call>')
assert len(bad) == 1 and bad[0]["name"] is None and "error" in bad[0], "malformed JSON must not raise"

reg = ToolRegistry()

@reg.register
def get_capacity(venue: str) -> int:
    """Number of people a venue in Lindenhof can hold."""
    return {"brunnen stadium": 3800, "town hall": 250, "observatory": 60}[venue.lower().strip()]

@reg.register
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME).eval()

# 2. What the model actually sees: the tool schemas are plain text inside the prompt.
prompt = tok.apply_chat_template([{"role": "user", "content": "What is 37 times 46?"}], tools=reg.schemas(), add_generation_prompt=True, tokenize=False)
assert '"name": "multiply"' in prompt and "<tool_call>" in prompt

# 3. Without tools the model has to compute by itself; with tools it asks for the function.
alone = generate_reply(model, tok, [{"role": "user", "content": "What is 37 times 46? Reply with the number only."}])
first, calls, final = tool_round(model, tok, reg, "What is 37 times 46?")
assert calls == [{"name": "multiply", "arguments": {"a": 37, "b": 46}}], calls
assert "1702" in final.replace(",", "")

first2, calls2, final2 = tool_round(model, tok, reg, "How many people does the Brunnen stadium hold?")
assert calls2 and calls2[0]["name"] == "get_capacity" and "3800" in final2.replace(",", "")

first3, calls3, final3 = tool_round(model, tok, reg, "Say hello in one word.")
assert calls3 == [] and first3 == final3, "a question that needs no tool should be answered directly"

print("m73_toolcall: all tests passed")
print("  the end of the system part of the prompt, as rendered by the chat template:")
for line in prompt.split("<|im_end|>")[0].split("\n")[-6:]:
    print("    " + line[:150])
print("  37 times 46 without tools:", repr(alone))
print("  first reply with tools:   ", repr(first))
print("  final answer:             ", repr(final))
