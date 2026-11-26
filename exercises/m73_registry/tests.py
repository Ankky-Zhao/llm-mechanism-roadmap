import json

reg = ToolRegistry()

@reg.register
def get_capacity(venue: str) -> int:
    """Number of people a venue in Lindenhof can hold.

    Only a few venues are known."""
    table = {"brunnen stadium": 3800, "town hall": 250, "observatory": 60}
    return table[venue.lower().strip()]

@reg.register
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@reg.register
def search(query: str, limit: int = 3, exact: bool = False) -> str:
    """Search the town archive."""
    return f"{limit} results for {query}"

assert get_capacity("Town Hall") == 250, "register must return the function unchanged"
assert list(reg.tools) == ["get_capacity", "multiply", "search"]

# 1. Schemas
assert reg.schema("get_capacity") == {"type": "function", "function": {"name": "get_capacity",
    "description": "Number of people a venue in Lindenhof can hold.",
    "parameters": {"type": "object", "properties": {"venue": {"type": "string"}}, "required": ["venue"]}}}
s = reg.schema("search")["function"]["parameters"]
assert s["properties"] == {"query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"}}
assert s["required"] == ["query"], "parameters with a default value are optional"
assert [x["function"]["name"] for x in reg.schemas()] == ["get_capacity", "multiply", "search"]
json.dumps(reg.schemas())                                     # must be serialisable, it is sent to the model as text

# 2. Validation
assert reg.validate("multiply", {"a": 3, "b": 2.5}) == []
assert reg.validate("launch_rocket", {}) == ["unknown tool: launch_rocket"]
assert reg.validate("multiply", "3, 4") == ["arguments must be an object"]
assert reg.validate("multiply", {"a": 3}) == ["missing argument: b"]
assert reg.validate("multiply", {"a": 3, "b": 4, "c": 5}) == ["unexpected argument: c"]
assert reg.validate("multiply", {"a": "3", "b": 4}) == ["argument a must be of type number"]
assert reg.validate("multiply", {"a": True, "b": 4}) == ["argument a must be of type number"], "a boolean is not a number"
assert reg.validate("search", {"query": "bridge", "limit": 2.5}) == ["argument limit must be of type integer"]
assert reg.validate("search", {"query": "bridge", "exact": True}) == []
assert len(reg.validate("search", {"limit": "many", "page": 2})) == 3

# 3. Calling never raises and always returns text
assert reg.call("multiply", {"a": 6, "b": 7}) == "42"
assert reg.call("get_capacity", {"venue": "Brunnen stadium"}) == "3800"
assert reg.call("search", {"query": "bridge"}) == '"3 results for bridge"'
assert reg.call("multiply", {"a": 6}) == "ERROR: missing argument: b"
assert reg.call("nope", {}) == "ERROR: unknown tool: nope"
out = reg.call("get_capacity", {"venue": "ice rink"})
assert out.startswith("ERROR: KeyError"), out
print("m73_registry: all tests passed")
print("  schema shown to the model:", json.dumps(reg.schema("search")))
