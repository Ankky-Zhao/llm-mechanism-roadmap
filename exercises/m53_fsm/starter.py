import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A deterministic finite automaton (DFA) over characters:
#   dfa = {"trans": {state: {char: next_state}}, "accept": set_of_states}      the start state is 0
# It describes which strings are allowed. The model produces tokens, and one token is usually several
# characters, so a token is allowed in a state if ALL of its characters can be followed in the DFA.


def dfa_walk(dfa, state, text):
    """Follow the characters of `text` from `state`. Returns the state reached, or None if some
    character has no transition (the text is not allowed here)."""
    # TODO 1
    raise NotImplementedError


def allowed_tokens(dfa, state, token_strings):
    """Ids of all tokens that may come next in `state`.

    token_strings[i] is the text of token i. A token is allowed if its text is not empty
    and dfa_walk does not return None.
    """
    # TODO 2
    raise NotImplementedError


@torch.no_grad()
def constrained_greedy(model, token_strings, dfa, input_ids, max_new_tokens, eos_id):
    """Greedy decoding in which only tokens allowed by the DFA can be chosen. Returns the generated text.

    At every step:
      1. allowed = allowed_tokens(dfa, state, token_strings); cache this per state, it is slow.
         If the state is accepting, the end-of-sequence token eos_id is allowed as well.
      2. Set the logits of all other tokens to -inf and take the argmax.
      3. If eos_id was chosen, stop. Otherwise append the token text and advance the state with dfa_walk.
      4. If the new state is accepting and has no outgoing transitions, stop: the output is complete.
    """
    # TODO 3
    raise NotImplementedError
