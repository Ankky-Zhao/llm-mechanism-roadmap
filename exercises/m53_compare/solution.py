import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_json(text):
    start, end = text.find("{"), text.find("}")
    if start == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def conforms(obj, schema):
    props = schema["properties"]
    if not isinstance(obj, dict) or set(obj) != set(props):
        return False
    for name, spec in props.items():
        value = obj[name]
        if "enum" in spec:
            ok = value in spec["enum"]
        elif spec["type"] == "boolean":
            ok = type(value) is bool
        elif spec["type"] == "integer":
            ok = type(value) is int                              # bool is a subclass of int in Python, so no isinstance
        else:
            ok = type(value) is str
        if not ok:
            return False
    return True


def format_error_rate(outputs, schema):
    bad = sum(1 for text in outputs if not conforms(extract_json(text), schema))
    return bad / len(outputs)


def field_accuracy(outputs, field, labels):
    hits = 0
    for text, label in zip(outputs, labels):
        obj = extract_json(text)
        hits += isinstance(obj, dict) and obj.get(field) == label
    return hits / len(outputs)
