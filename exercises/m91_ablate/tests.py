import random
import torch

model, tok = load_gpt2()
N = 20
seqs = [repeated_tokens(50257, N, tok.bos_token_id, seed=s) for s in range(5)]

# 1. token_losses agrees with the loss computed by the model.
ids = seqs[0]
with torch.no_grad():
    ref = model(ids[None], labels=ids[None]).loss.item()
tl = token_losses(model, ids)
assert tl.shape == (2 * N,)
assert abs(tl.mean().item() - ref) < 1e-4

# 2. Ablating nothing changes nothing, and the hooks are removed afterwards.
with torch.no_grad():
    base_logits = model(ids[None]).logits
    with ablate_heads(model, []):
        assert torch.equal(model(ids[None]).logits, base_logits)
    with ablate_heads(model, [(5, 5)]):
        changed = model(ids[None]).logits
    assert not torch.allclose(changed, base_logits, atol=1e-3), "ablating a head changes the output"
    assert torch.equal(model(ids[None]).logits, base_logits), "the hooks have to be removed on exit"
assert all(len(b.attn.c_proj._forward_pre_hooks) == 0 for b in model.transformer.h)

# 3. Ablating a head equals zeroing its slice by hand.
with torch.no_grad():
    captured = {}
    def grab(module, args): captured["x"] = args[0].clone()
    with ablate_heads(model, [(5, 1), (5, 5)]):
        h0 = model.transformer.h[5].attn.c_proj.register_forward_pre_hook(grab)     # runs after the ablation hook
        model(ids[None])
        h0.remove()
x = captured["x"]
assert x[..., 64:128].abs().max() == 0 and x[..., 320:384].abs().max() == 0, "heads 1 and 5 of layer 5 are zero"
assert x[..., :64].abs().max() > 0 and x[..., 128:320].abs().max() > 0, "the other heads are untouched"

# 4. The causal test. Induction heads are found from attention, their role is tested by removing them.
scores = torch.stack([induction_score(get_attention(model, s), N) for s in seqs]).mean(0)
ind = top_heads(scores, 6)
mean_loss = lambda heads: sum(second_half_loss_with(heads, s) for s in seqs) / len(seqs)
def second_half_loss_with(heads, s):
    with ablate_heads(model, heads):
        return second_half_loss(model, s, N)
base = mean_loss([])
abl3 = mean_loss(ind[:3])
abl6 = mean_loss(ind)
rng = random.Random(0)
pool = [(l, h) for l in range(12) for h in range(12) if (l, h) not in ind]
rand6 = [mean_loss(rng.sample(pool, 6)) for _ in range(10)]
assert abl6 > abl3 > base
assert abl6 > base + 2.0, "removing the induction heads destroys most of the in-context prediction"
assert abl6 > max(rand6), "more than removing any of 10 random sets of 6 other heads"

print(f"second-half loss, mean over {len(seqs)} sequences")
print(f"  intact model                      {base:.2f}")
print(f"  top 3 induction heads removed     {abl3:.2f}")
print(f"  top 6 induction heads removed     {abl6:.2f}   heads: " + " ".join(f"L{l}H{h}" for l, h in ind))
print(f"  6 random other heads removed      {sum(rand6) / len(rand6):.2f} on average, range {min(rand6):.2f} to {max(rand6):.2f} over 10 draws")

# Attention weight and causal effect are not the same thing.
prev = (4, 11)
single = {hd: mean_loss([hd]) - base for hd in ind + [prev]}
for (l, h), d in single.items():
    kind = "previous-token" if (l, h) == prev else "induction"
    print(f"  L{l}H{h} alone ({kind}, score {scores[l, h]:.2f}): loss change {d:+.2f}")

# The same heads on ordinary text.
text = "The Eiffel Tower is in Paris. It was built for the 1889 World's Fair and was the tallest structure in the world until 1930."
tids = tok(text, return_tensors="pt").input_ids[0]
t0 = token_losses(model, tids).mean().item()
with ablate_heads(model, ind):
    t1 = token_losses(model, tids).mean().item()
print(f"ordinary text: loss {t0:.2f} intact, {t1:.2f} with the 6 induction heads removed")
