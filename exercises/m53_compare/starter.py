import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# dfa_walk, allowed_tokens, constrained_greedy (exercise 1) and compile_schema (exercise 2) are defined above.


def extract_json(text):
    """The first {...} in `text`, parsed. Returns the object, or None if there is none or it does not parse.

    Take the text from the first "{" up to and including the first "}" (the schemas here are not nested)
    and pass it to json.loads; catch json.JSONDecodeError.
    """
    # TODO 1
    raise NotImplementedError


def conforms(obj, schema):
    """True if obj is a dict with exactly the properties of the schema and every value has the right type.

    The value kinds are those of exercise 2: {"enum": [...]}, boolean, integer, string.
    Careful: in Python, isinstance(True, int) is True. Compare with type(value) is int.
    """
    # TODO 2
    raise NotImplementedError


def format_error_rate(outputs, schema):
    """Fraction of output texts that contain no JSON object conforming to the schema."""
    # TODO 3
    raise NotImplementedError


def field_accuracy(outputs, field, labels):
    """Fraction of outputs whose parsed object has obj[field] == label. Unparseable outputs count as wrong."""
    # TODO 4
    raise NotImplementedError
