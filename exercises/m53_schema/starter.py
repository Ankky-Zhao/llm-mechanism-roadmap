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


# Every function below takes the Builder and the set of states reached so far, adds transitions,
# and returns the set of states in which its piece of text can end.


def literal(b, states, text):
    """A fixed piece of text, one b.step per character."""
    # TODO 1
    raise NotImplementedError


def choice(b, states, options):
    """One of several fixed strings, for example ["true", "false"].

    Options may share a prefix ("red", "rose"), so build a trie: for each option walk from the start
    state, follow a transition if the character already has one, otherwise create a new state with
    b.new_state() and add the transition to b.trans directly. Returns the set of end states.
    `states` always contains exactly one state here.
    """
    # TODO 2
    raise NotImplementedError


def integer(b, states, max_digits=4):
    """A non-negative integer without leading zeros: "0", or a digit 1-9 followed by up to
    max_digits - 1 further digits. Returns all states in which the number may end."""
    # TODO 3
    raise NotImplementedError


def free_string(b, states, max_length):
    """A JSON string of at most max_length characters from STRING_CHARS, including both quotes.

    The closing quote may come after 0, 1, ..., max_length characters and leads to one shared state.
    """
    # TODO 4
    raise NotImplementedError


def compile_schema(schema):
    """Compile a small subset of JSON Schema into a DFA {"trans": ..., "accept": ...}.

    Supported: an object whose properties are all required and appear in the given order, with values
      {"enum": [strings]}, {"type": "boolean"}, {"type": "integer", "maxDigits": n}, {"type": "string", "maxLength": n}.
    The output format is fixed:  {"name": value, "name2": value}
    (one space after each colon and comma; json.dumps(name) gives the quoted key).
    """
    # TODO 5
    raise NotImplementedError
