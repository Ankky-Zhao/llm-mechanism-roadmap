import torch


def load_bridge():
    """GPT-2 small wrapped by TransformerLens 4. Requires `pip install transformer_lens`."""
    from transformer_lens import TransformerBridge
    return TransformerBridge.boot_transformers("gpt2")


@torch.no_grad()
def tl_residual_patching_map(bridge, clean_ids, corrupt_ids, ans, wrong):
    """The residual patching map of exercise 1, written with the TransformerLens interface."""
    clean_logits, cache = bridge.run_with_cache(clean_ids)
    corrupt_logits = bridge(corrupt_ids)
    c, x = logit_diff(clean_logits, ans, wrong), logit_diff(corrupt_logits, ans, wrong)
    L, T = bridge.cfg.n_layers, clean_ids.shape[1]
    out = torch.zeros(L, T)
    for l in range(L):
        name = f"blocks.{l}.hook_resid_post"
        for p in range(T):
            def hook(act, hook, p=p):
                act[:, p] = cache[hook.name][:, p]
                return act
            patched = bridge.run_with_hooks(corrupt_ids, fwd_hooks=[(name, hook)])
            out[l, p] = normalized(logit_diff(patched, ans, wrong), c, x)
    return out
