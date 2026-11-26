import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A deterministic finite automaton (DFA) over characters:
#   dfa = {"trans": {state: {char: next_state}}, "accept": set_of_states}      the start state is 0


def dfa_walk(dfa, state, text):
    for ch in text:
        state = dfa["trans"].get(state, {}).get(ch)
        if state is None:
            return None
    return state


def allowed_tokens(dfa, state, token_strings):
    return [i for i, s in enumerate(token_strings) if s and dfa_walk(dfa, state, s) is not None]


@torch.no_grad()
def constrained_greedy(model, token_strings, dfa, input_ids, max_new_tokens, eos_id):
    state, text, cache = 0, "", {}
    for _ in range(max_new_tokens):
        if state not in cache:
            cache[state] = allowed_tokens(dfa, state, token_strings)
        allowed = list(cache[state])
        if state in dfa["accept"]:
            allowed.append(eos_id)                              # a complete output may stop here
        if not allowed:
            break
        logits = model(input_ids).logits[0, -1]
        mask = torch.full_like(logits, float("-inf"))
        mask[allowed] = 0.0
        next_id = (logits + mask).argmax().item()
        if next_id == eos_id and state in dfa["accept"]:
            break
        text += token_strings[next_id]
        state = dfa_walk(dfa, state, token_strings[next_id])
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        if state in dfa["accept"] and not dfa["trans"].get(state):
            break                                               # nothing can follow, so the output is finished
    return text
