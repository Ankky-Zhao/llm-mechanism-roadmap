import json

STRING_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'"
DIGITS = "0123456789"


class Builder:
    """Builds a DFA step by step. Nothing to change in this class."""

    def __init__(self):
        self.trans = {0: {}}

    def new_state(self):
        s = len(self.trans)
        self.trans[s] = {}
        return s

    def step(self, states, chars):
        """From every state in `states`, each character in `chars` leads to one new shared state.
        Returns that state as a one-element set."""
        target = self.new_state()
        for s in states:
            for ch in chars:
                assert ch not in self.trans[s], f"state {s} already has a transition on {ch!r}"
                self.trans[s][ch] = target
        return {target}


def literal(b, states, text):
    for ch in text:
        states = b.step(states, ch)
    return states


def choice(b, states, options):
    (start,) = states                                      # a choice always starts from a single state
    ends = set()
    for option in options:
        s = start
        for ch in option:                                  # walk the shared prefix, branch where it ends
            if ch not in b.trans[s]:
                b.trans[s][ch] = b.new_state()
            s = b.trans[s][ch]
        ends.add(s)
    return ends


def integer(b, states, max_digits=4):
    zero = b.step(states, "0")                             # "0" alone, no leading zeros otherwise
    cur = b.step(states, "123456789")
    ends = zero | cur
    for _ in range(max_digits - 1):
        cur = b.step(cur, DIGITS)
        ends |= cur
    return ends


def free_string(b, states, max_length):
    cur = b.step(states, '"')
    inside = set(cur)                                      # after 0 characters
    for _ in range(max_length):
        cur = b.step(cur, STRING_CHARS)
        inside |= cur
    return b.step(inside, '"')                             # the closing quote is possible after any length


def compile_schema(schema):
    b = Builder()
    states = literal(b, {0}, "{")
    for i, (name, spec) in enumerate(schema["properties"].items()):
        states = literal(b, states, (", " if i else "") + json.dumps(name) + ": ")
        if "enum" in spec:
            states = choice(b, states, [json.dumps(v) for v in spec["enum"]])
        elif spec["type"] == "boolean":
            states = choice(b, states, ["true", "false"])
        elif spec["type"] == "integer":
            states = integer(b, states, spec.get("maxDigits", 4))
        elif spec["type"] == "string":
            states = free_string(b, states, spec.get("maxLength", 20))
        else:
            raise ValueError(f"unsupported: {spec}")
    states = literal(b, states, "}")
    return {"trans": b.trans, "accept": set(states)}
