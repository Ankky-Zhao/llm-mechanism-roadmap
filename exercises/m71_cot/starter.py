import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text):
    """The final numeric answer in a model reply, as a float, or None.

    If the text contains "Answer: <number>" (possibly with ** or $ around the number), take the LAST such number.
    Otherwise take the last number anywhere in the text. Remove thousands separators (1,234 -> 1234).
    A pattern for one number:  -?\\d[\\d,]*(?:\\.\\d+)?
    """
    # TODO 1
    raise NotImplementedError


@torch.no_grad()
def ask(model, tokenizer, question, mode, max_new_tokens=180):
    """Ask a chat model in one of two ways and return its reply as text.

    mode "direct": the user message is the question itself, and the reply is forced to begin with
                   "Answer:" by appending that text after the chat template (a "prefill").
                   Generate at most 8 tokens. Return "Answer:" + the generated text.
    mode "cot":    the user message is the question followed by
                   " Think step by step. End with a line of the form 'Answer: <number>'."
                   Generate up to max_new_tokens tokens. Return the generated text.

    Build the prompt as text with
        tokenizer.apply_chat_template([{"role": "user", "content": ...}], add_generation_prompt=True, tokenize=False)
    tokenise it, and decode only the newly generated tokens with skip_special_tokens=True. Use greedy decoding.
    """
    # TODO 2
    raise NotImplementedError
