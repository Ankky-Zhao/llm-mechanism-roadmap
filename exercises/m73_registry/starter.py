import inspect
import json

# A model cannot run code. It can only write text that describes which function it wants called and with
# which arguments. The program around the model needs three things for that: a description of every tool
# that can be shown to the model, a check of the arguments the model wrote, and a safe way to run the tool.

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


class ToolRegistry:
    def __init__(self):
        self.tools = {}                                        # name -> Python function

    def register(self, fn):
        """Store the function under fn.__name__ and return it unchanged (so that @registry.register works)."""
        # TODO 1
        raise NotImplementedError

    def schema(self, name):
        """The description of one tool in the JSON format that chat templates and APIs expect:

        {"type": "function", "function": {"name": ..., "description": ...,
            "parameters": {"type": "object", "properties": {param: {"type": ...}}, "required": [...]}}}

        Build it from the function itself with inspect.signature(fn).parameters:
          the type of a parameter is JSON_TYPES[p.annotation];
          a parameter is required if it has no default (p.default is inspect.Parameter.empty);
          the description is the first line of the docstring.
        """
        # TODO 2
        raise NotImplementedError

    def schemas(self):
        """The schemas of all registered tools, in registration order."""
        # TODO 3
        raise NotImplementedError

    def validate(self, name, arguments):
        """Check a call the model asked for. Returns a list of error strings, empty if the call is fine.

          unknown tool                 -> ["unknown tool: <name>"]            (and nothing else)
          arguments is not a dict      -> ["arguments must be an object"]
          a required argument missing  -> "missing argument: <param>"
          an argument not in the schema-> "unexpected argument: <key>"
          wrong type                   -> "argument <key> must be of type <type>"
        A JSON integer is acceptable where a number is expected; a boolean is never a number
        (in Python, True is an int, so compare with type(value) is ...).
        """
        # TODO 4
        raise NotImplementedError

    def call(self, name, arguments):
        """Validate, run, and ALWAYS return a string, because the result goes back to the model as text.

        Invalid call:    "ERROR: " + the errors joined by "; "
        The tool raises: "ERROR: <ExceptionName>: <message>"
        Otherwise:       json.dumps(result)
        """
        # TODO 5
        raise NotImplementedError
