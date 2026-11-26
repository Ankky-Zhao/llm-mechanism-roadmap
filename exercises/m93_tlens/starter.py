import torch


def load_bridge():
    """GPT-2 small wrapped by TransformerLens 4. Requires `pip install transformer_lens`."""
    from transformer_lens import TransformerBridge
    return TransformerBridge.boot_transformers("gpt2")


@torch.no_grad()
def tl_residual_patching_map(bridge, clean_ids, corrupt_ids, ans, wrong):
    """The residual patching map of exercise 1, written with the TransformerLens interface."""
    # TODO 1: clean_logits, cache = bridge.run_with_cache(clean_ids). corrupt_logits = bridge(corrupt_ids)
    # TODO 2: the output of block l is stored under the name f"blocks.{l}.hook_resid_post"
    # TODO 3: a hook has the signature hook(act, hook) and returns the new activation. hook.name is the
    #         name it was attached to. Run with bridge.run_with_hooks(corrupt_ids, fwd_hooks=[(name, fn)]),
    #         which removes the hook again by itself.
    raise NotImplementedError
