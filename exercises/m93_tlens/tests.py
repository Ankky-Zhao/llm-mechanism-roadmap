import time
import torch

model, tok = load_gpt2()
clean, corrupt, ans, wrong = ioi_batch(tok)
bridge = load_bridge()

# 1. The wrapped model computes the same logits and the same residual stream as the Hugging Face model.
tl_logits, tl_cache = bridge.run_with_cache(clean)
hf_logits, hf_cache = run_with_cache(model, clean)
assert torch.allclose(tl_logits, hf_logits, atol=1e-3)
for l in (0, 5, 11):
    assert torch.allclose(tl_cache[f"blocks.{l}.hook_resid_post"], hf_cache[("resid", l)], rtol=1e-4, atol=1e-2), l
z = tl_cache["blocks.8.attn.hook_z"]
assert z.shape == (8, 14, 12, 64), "TransformerLens stores the head outputs with a separate head axis"
assert torch.allclose(z.reshape(8, 14, 768), hf_cache[("z", 8)], rtol=1e-4, atol=1e-3)

# 2. The two implementations of the patching map agree.
t0 = time.time(); mine = residual_patching_map(model, clean, corrupt, ans, wrong); t_mine = time.time() - t0
t0 = time.time(); theirs = tl_residual_patching_map(bridge, clean, corrupt, ans, wrong); t_tl = time.time() - t0
assert theirs.shape == (12, 14)
assert torch.allclose(mine, theirs, atol=1e-2), (mine - theirs).abs().max()

print(f"largest difference between the two maps: {(mine - theirs).abs().max().item():.5f}")
print(f"raw hooks {t_mine:.1f} s, TransformerLens {t_tl:.1f} s")
print(f"number of hook points TransformerLens exposes on GPT-2 small: {len(tl_cache)}")
print("hook points of one block:", ", ".join(k.split('blocks.8.')[1] for k in tl_cache if k.startswith("blocks.8.hook_")))
