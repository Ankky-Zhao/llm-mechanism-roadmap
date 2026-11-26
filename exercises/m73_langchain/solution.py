import json
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool


def interface(schema):
    f = schema["function"]
    params = f["parameters"]
    return {"name": f["name"],
            "types": {k: v.get("type") for k, v in params.get("properties", {}).items()},
            "required": sorted(params.get("required", []))}


class LangChainRegistry:
    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}

    def schemas(self):
        return [convert_to_openai_tool(t) for t in self.tools.values()]

    def call(self, name, arguments):
        if name not in self.tools:
            return f"ERROR: unknown tool: {name}"
        try:
            return json.dumps(self.tools[name].invoke(arguments))
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {str(e)[:200]}"
