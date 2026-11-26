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
    last = logits[:, -1]
    return (last.gather(1, ans[:, None]) - last.gather(1, wrong[:, None]))[:, 0]


def block_output(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def run_with_cache(model, ids):
    """Run the model and store activations. Returns (logits, cache).

    cache[("resid", l)]: output of block l, shape (N, T, 768)
    cache[("z", l)]:     input of attn.c_proj of block l, the concatenated head outputs, (N, T, 768)
    """
    cache, handles = {}, []
    for l, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(
            lambda m, a, out, l=l: cache.__setitem__(("resid", l), block_output(out).clone())))
        handles.append(block.attn.c_proj.register_forward_pre_hook(
            lambda m, a, l=l: cache.__setitem__(("z", l), a[0].clone())))
    try:
        logits = model(ids).logits
    finally:
        for h in handles:
            h.remove()
    return logits, cache


def normalized(patched, clean, corrupt):
    """0 if the patched run behaves like the corrupted run, 1 if like the clean run. Tensors (N,) -> float."""
    return ((patched.mean() - corrupt.mean()) / (clean.mean() - corrupt.mean())).item()


@torch.no_grad()
def patch_residual(model, corrupt_ids, clean_cache, layer, pos):
    """Run on the corrupted input with the output of block `layer` at position `pos` replaced by its
    clean value. pos is an int or slice(None). Returns the logits (N, T, V)."""
    def hook(module, args, out):
        x = block_output(out).clone()
        x[:, pos] = clean_cache[("resid", layer)][:, pos]
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
    handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        return model(corrupt_ids).logits
    finally:
        handle.remove()


def residual_patching_map(model, clean_ids, corrupt_ids, ans, wrong):
    """Normalized effect of patching every (layer, position). Returns a tensor (n_layer, T)."""
    clean_logits, cache = run_with_cache(model, clean_ids)
    with torch.no_grad():
        corrupt_logits = model(corrupt_ids).logits
    c, x = logit_diff(clean_logits, ans, wrong), logit_diff(corrupt_logits, ans, wrong)
    L, T = model.config.n_layer, clean_ids.shape[1]
    out = torch.zeros(L, T)
    for l in range(L):
        for p in range(T):
            out[l, p] = normalized(logit_diff(patch_residual(model, corrupt_ids, cache, l, p), ans, wrong), c, x)
    return out
