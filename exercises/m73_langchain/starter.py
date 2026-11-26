import json
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

# ToolRegistry (exercise 1), parse_tool_calls (exercise 2) and run_agent (exercise 3) are defined above.
# LangChain offers the same pieces ready-made:  @tool turns a function into a tool object with a name, a
# description and an argument schema;  convert_to_openai_tool(tool_object) gives the JSON schema;
# tool_object.invoke({...}) validates the arguments and runs the function.


def interface(schema):
    """Reduce a tool schema to what matters for comparing two libraries:

    {"name": ..., "types": {param: json_type}, "required": sorted list of required params}
    Both the hand-written registry and LangChain produce {"type": "function", "function": {...}};
    LangChain adds extra keys (titles, descriptions per parameter), which this function ignores.
    """
    # TODO 1
    raise NotImplementedError


class LangChainRegistry:
    """An adapter that gives LangChain tools the two methods run_agent needs: schemas() and call()."""

    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}

    def schemas(self):
        # TODO 2: convert_to_openai_tool for every tool
        raise NotImplementedError

    def call(self, name, arguments):
        """Same contract as ToolRegistry.call: always a string, errors as "ERROR: ..." text.
        Unknown tool -> "ERROR: unknown tool: <name>". Otherwise json.dumps(tool.invoke(arguments));
        if invoke raises (LangChain raises on invalid arguments), return "ERROR: <ExceptionName>: <message>".
        """
        # TODO 3
        raise NotImplementedError
