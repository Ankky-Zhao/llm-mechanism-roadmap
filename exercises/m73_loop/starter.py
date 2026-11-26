import json

# ToolRegistry (exercise 1), parse_tool_calls and generate_reply (exercise 2) are defined above.


def run_agent(generate, registry, messages, max_steps=5):
    """The agent loop: let the model speak, run the tools it asks for, show it the results, repeat.

    generate:  a function messages -> reply text (the model, or a scripted stand-in in the tests)
    registry:  anything with .call(name, arguments) -> str
    messages:  the conversation so far; work on a copy, do not modify the list passed in
    Returns (answer, trace). trace is a list of {"name", "arguments", "result"}, one per tool call.

    In each step:
      1. reply = generate(messages); calls = parse_tool_calls(reply)
      2. no calls  -> return (reply, trace)
      3. append the assistant turn with its tool_calls (as in exercise 2; leave out calls that failed to parse)
      4. for every call append a tool message with the result of registry.call. For a call that failed to
         parse, the result is "ERROR: " + its error text and the tool name is "unknown".
         Errors are not raised: the model reads them and can react.
    After max_steps steps without a final answer return (None, trace).
    """
    # TODO 1
    raise NotImplementedError
