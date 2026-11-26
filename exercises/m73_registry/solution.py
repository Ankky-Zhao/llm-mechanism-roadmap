import inspect
import json

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, fn):
        self.tools[fn.__name__] = fn
        return fn                                              # so that it can be used as a decorator

    def schema(self, name):
        fn = self.tools[name]
        properties, required = {}, []
        for pname, p in inspect.signature(fn).parameters.items():
            properties[pname] = {"type": JSON_TYPES[p.annotation]}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        description = (fn.__doc__ or "").strip().split("\n")[0]
        return {"type": "function", "function": {"name": name, "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required}}}

    def schemas(self):
        return [self.schema(name) for name in self.tools]

    def validate(self, name, arguments):
        if name not in self.tools:
            return [f"unknown tool: {name}"]
        if not isinstance(arguments, dict):
            return ["arguments must be an object"]
        params = self.schema(name)["function"]["parameters"]
        errors = [f"missing argument: {p}" for p in params["required"] if p not in arguments]
        for key, value in arguments.items():
            if key not in params["properties"]:
                errors.append(f"unexpected argument: {key}")
                continue
            expected = params["properties"][key]["type"]
            ok = {"string": type(value) is str, "boolean": type(value) is bool, "integer": type(value) is int,
                  "number": type(value) in (int, float)}[expected]
            if not ok:
                errors.append(f"argument {key} must be of type {expected}")
        return errors

    def call(self, name, arguments):
        errors = self.validate(name, arguments)
        if errors:
            return "ERROR: " + "; ".join(errors)
        try:
            return json.dumps(self.tools[name](**arguments))
        except Exception as e:                                  # a failing tool must not crash the agent
            return f"ERROR: {type(e).__name__}: {e}"
