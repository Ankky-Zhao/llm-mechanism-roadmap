import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gpt2():
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


PAIRS = [(" Mary", " John"), (" Tom", " James"), (" Sarah", " David"), (" Paul", " Emma"),
         (" John", " Mary"), (" James", " Tom"), (" David", " Sarah"), (" Emma", " Paul")]
TEMPLATE = "When{a} and{b} went to the store,{s} gave a drink to"
S2, END = 9, 13          # positions of the repeated name and of the last token


def ioi_batch(tok):
    """Eight prompt pairs of the indirect object identification task, all 14 tokens long.

    clean:   When A and B went to the store, B gave a drink to   -> A
    corrupt: When A and B went to the store, A gave a drink to   -> B
    The two differ at position S2 only. Returns (clean, corrupt, ans, wrong): ids (8, 14) twice and
    the token ids of A and B, each (8,).
    """
    enc = lambda s: tok(s, return_tensors="pt").input_ids[0]
    clean = torch.stack([enc(TEMPLATE.format(a=a, b=b, s=b)) for a, b in PAIRS])
    corrupt = torch.stack([enc(TEMPLATE.format(a=a, b=b, s=a)) for a, b in PAIRS])
    ans = torch.tensor([tok(a).input_ids[0] for a, b in PAIRS])
    wrong = torch.tensor([tok(b).input_ids[0] for a, b in PAIRS])
    return clean, corrupt, ans, wrong


def logit_diff(logits, ans, wrong):
    """logit(ans) - logit(wrong) at the last position. logits (N, T, V) -> (N,)."""
    # TODO 1: take the last position, then gather the logits of ans and of wrong and subtract
    raise NotImplementedError


def block_output(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def run_with_cache(model, ids):
    """Run the model and store activations. Returns (logits, cache).

    cache[("resid", l)]: output of block l, shape (N, T, 768)
    cache[("z", l)]:     input of attn.c_proj of block l, the concatenated head outputs, (N, T, 768)
    """
    cache, handles = {}, []
    # TODO 2: for every block l register
    #   a forward hook on the block that stores block_output(out).clone() under ("resid", l)
    #   a forward pre-hook on block.attn.c_proj that stores args[0].clone() under ("z", l)
    #   Bind l in the lambda with a default argument (lambda m, a, out, l=l: ...).
    # TODO 3: run the model, remove the handles in a finally clause, return (logits, cache)
    raise NotImplementedError


def normalized(patched, clean, corrupt):
    """0 if the patched run behaves like the corrupted run, 1 if like the clean run. Tensors (N,) -> float."""
    # TODO 4: (mean patched - mean corrupt) / (mean clean - mean corrupt), as a Python float
    raise NotImplementedError


@torch.no_grad()
def patch_residual(model, corrupt_ids, clean_cache, layer, pos):
    """Run on the corrupted input with the output of block `layer` at position `pos` replaced by its
    clean value. pos is an int or slice(None). Returns the logits (N, T, V)."""
    # TODO 5: a forward hook on model.transformer.h[layer] that copies block_output(out), overwrites
    #         [:, pos] with the clean value and returns the result. If out is a tuple, return a tuple
    #         with the new tensor in first place: (x,) + tuple(out[1:]).
    # TODO 6: run on corrupt_ids, remove the handle, return the logits
    raise NotImplementedError


def residual_patching_map(model, clean_ids, corrupt_ids, ans, wrong):
    """Normalized effect of patching every (layer, position). Returns a tensor (n_layer, T)."""
    # TODO 7: cache the clean run, compute the logit differences of the clean and the corrupted run,
    #         then loop over layers and positions and fill a tensor (n_layer, T) with normalized(...)
    raise NotImplementedError
