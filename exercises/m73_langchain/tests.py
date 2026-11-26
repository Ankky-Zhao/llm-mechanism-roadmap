import json
from langchain_core.tools import tool

# The same two tools, written once for the hand-made registry and once for LangChain.
mine = ToolRegistry()

@mine.register
def get_capacity(venue: str) -> int:
    """Number of people a venue in Lindenhof can hold."""
    return {"brunnen stadium": 3800, "town hall": 250, "observatory": 60}[venue.lower().strip()]

@mine.register
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def get_capacity(venue: str) -> int:
    """Number of people a venue in Lindenhof can hold."""
    return {"brunnen stadium": 3800, "town hall": 250, "observatory": 60}[venue.lower().strip()]

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

theirs = LangChainRegistry([get_capacity, multiply])

# 1. Both describe the tools to the model in the same way.
a, b = mine.schemas(), theirs.schemas()
assert [interface(x) for x in a] == [interface(x) for x in b], "name, parameter types and required parameters should agree"
assert interface(a[1]) == {"name": "multiply", "types": {"a": "number", "b": "number"}, "required": ["a", "b"]}
assert b[0]["function"]["description"] == a[0]["function"]["description"]

# 2. Both run the tools and report problems as text.
assert float(theirs.call("multiply", {"a": 6, "b": 7})) == float(mine.call("multiply", {"a": 6, "b": 7})) == 42
assert theirs.call("get_capacity", {"venue": "Town hall"}) == "250"
assert theirs.call("nope", {}) == "ERROR: unknown tool: nope"
assert theirs.call("multiply", {"a": 6}).startswith("ERROR"), "a missing argument is rejected by LangChain's own validation"
assert theirs.call("get_capacity", {"venue": "ice rink"}).startswith("ERROR: KeyError")

# 3. The loop from exercise 3 runs unchanged on either registry.
call = lambda name, **args: "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"
def scripted(replies):
    it = iter(replies)
    return lambda messages: next(it)
script = [call("get_capacity", venue="Brunnen stadium"), call("multiply", a=3800, b=12), "That is 45600 euros."]
ans1, tr1 = run_agent(scripted(script), mine, [{"role": "user", "content": "q"}])
ans2, tr2 = run_agent(scripted(script), theirs, [{"role": "user", "content": "q"}])
assert ans1 == ans2 and [t["name"] for t in tr1] == [t["name"] for t in tr2]
assert [float(t["result"]) for t in tr1] == [float(t["result"]) for t in tr2] == [3800.0, 45600.0]

print("m73_langchain: all tests passed")
print("  hand-written schema:", json.dumps(a[1]))
print("  LangChain schema:   ", json.dumps(b[1]))
