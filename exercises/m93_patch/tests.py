import torch

model, tok = load_gpt2()
clean, corrupt, ans, wrong = ioi_batch(tok)
assert clean.shape == corrupt.shape == (8, 14)
assert (clean != corrupt).sum(0).tolist() == [0] * 9 + [8] + [0] * 4, "the pairs differ at position 9 only"

# 1. Logit difference on a hand-made tensor.
fake = torch.zeros(2, 3, 5); fake[0, -1] = torch.tensor([1.0, 4.0, 0, 0, 0]); fake[1, -1] = torch.tensor([0, 0, 2.0, 0, 7.0])
assert logit_diff(fake, torch.tensor([1, 2]), torch.tensor([0, 4])).tolist() == [3.0, -5.0]
assert abs(normalized(torch.tensor([1.0, 3.0]), torch.tensor([4.0, 4.0]), torch.tensor([0.0, 0.0])) - 0.5) < 1e-6

# 2. The cache holds what the model computes, and no hook is left behind.
hooks_now = lambda: sum(len(b._forward_hooks) + len(b.attn.c_proj._forward_pre_hooks) for b in model.transformer.h)
clean_logits, cache = run_with_cache(model, clean)
assert hooks_now() == 0, "remove the hooks after the run"
with torch.no_grad():
    ref = model(clean, output_hidden_states=True)
assert torch.allclose(clean_logits, ref.logits, atol=1e-4)
assert len(cache) == 24 and cache[("resid", 5)].shape == (8, 14, 768) and cache[("z", 5)].shape == (8, 14, 768)
for l in range(11):
    assert torch.allclose(cache[("resid", l)], ref.hidden_states[l + 1], rtol=1e-4, atol=1e-2), l
n_hooks = hooks_now()          # transformers itself may keep hooks on the blocks after output_hidden_states

# 3. The model solves the task on the clean prompts and is misled on the corrupted ones.
with torch.no_grad():
    corrupt_logits = model(corrupt).logits
c, x = logit_diff(clean_logits, ans, wrong), logit_diff(corrupt_logits, ans, wrong)
assert (c > 0).all() and (x < 0).all()

# 4. Patching all positions of a layer turns the corrupted run into the clean run.
full = patch_residual(model, corrupt, cache, 6, slice(None))
assert torch.allclose(full, clean_logits, atol=1e-3)
assert hooks_now() == n_hooks
# Patching a position at which the two inputs agree changes nothing.
same = patch_residual(model, corrupt, cache, 6, 3)
assert torch.allclose(same, corrupt_logits, atol=1e-4)

# 5. The full map.
M = residual_patching_map(model, clean, corrupt, ans, wrong)
assert M.shape == (12, 14)
assert M[:, :S2].abs().max() < 1e-3, "positions before the changed token carry no difference"
assert M[0, S2] > 0.95, "in layer 0 the whole difference sits at the changed token"
assert abs(M[11, END] - 1) < 1e-3, "after the last block only the last position matters"
assert M[11, S2] < 0.01
cross = min(l for l in range(12) if M[l, END] > M[l, S2])
assert 6 <= cross <= 9, cross

print(f"logit difference: clean {c.mean():.2f}, corrupted {x.mean():.2f} (mean over 8 pairs)")
tokens = [tok.decode(t) for t in clean[0]]
print("normalized effect of patching the residual stream after each block, at the positions from S2 on")
print("layer  " + "".join(f"{t!r:>9}" for t in tokens[S2:]))
for l in range(12):
    print(f"{l:>5}  " + "".join(f"{M[l, p].item():>9.2f}" for p in range(S2, 14)))
print(f"the information moves from position {S2} to the last position between layers {cross - 1} and {cross}")
