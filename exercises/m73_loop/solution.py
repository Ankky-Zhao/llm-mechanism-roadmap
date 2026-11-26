import json


def run_agent(generate, registry, messages, max_steps=5):
    messages = list(messages)
    trace = []
    for _ in range(max_steps):
        reply = generate(messages)
        calls = parse_tool_calls(reply)
        if not calls:
            return reply, trace                                  # no tool requested: this is the final answer
        valid = [c for c in calls if c["name"] is not None]
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in valid]})
        for c in calls:
            if c["name"] is None:
                name, result = "unknown", "ERROR: " + c["error"]
            else:
                name, result = c["name"], registry.call(c["name"], c["arguments"])
            trace.append({"name": c["name"], "arguments": c["arguments"], "result": result})
            messages.append({"role": "tool", "name": name, "content": result})
    return None, trace                                           # step limit reached without a final answer
