import torch

# 1. Hand-made patterns.
T = 6
prev = torch.zeros(T, T); prev[0, 0] = 1
for i in range(1, T): prev[i, i - 1] = 1
self_ = torch.eye(T)
fake = torch.stack([prev, self_])[None]                    # (1, 2, T, T)
s = prev_token_score(fake)
assert s.shape == (1, 2)
assert torch.allclose(s, torch.tensor([[1.0, 0.0]]))

# 2. The repeated sequence.
seq = repeated_tokens(50257, 20, 50256, seed=0)
assert seq.shape == (41,) and seq[0] == 50256
assert torch.equal(seq[1:21], seq[21:41])
assert torch.equal(seq, repeated_tokens(50257, 20, 50256, seed=0))
assert not torch.equal(seq, repeated_tokens(50257, 20, 50256, seed=1))

# 3. A hand-made induction pattern for n = 3: positions 4, 5, 6 read positions 2, 3, 4.
n = 3
ind = torch.zeros(7, 7)
for i in range(0, 4): ind[i, 0] = 1
for i in range(4, 7): ind[i, i - n + 1] = 1
copy = torch.zeros(7, 7)
for i in range(0, 4): copy[i, 0] = 1
for i in range(4, 7): copy[i, i - n] = 1                    # reads the earlier copy of the same token instead
s = induction_score(torch.stack([ind, copy])[None], n)
assert torch.allclose(s, torch.tensor([[1.0, 0.0]]))
assert top_heads(torch.tensor([[0.1, 0.9], [0.5, 0.3]]), 2) == [(0, 1), (1, 0)]

# 4. GPT-2 small.
model, tok = load_gpt2()
text = "The cat sat on the mat because it was tired and the dog went to sleep in the garden"
att = get_attention(model, tok(text, return_tensors="pt").input_ids[0])
ps = prev_token_score(att)
best_prev = top_heads(ps, 3)
assert best_prev[0] == (4, 11), best_prev
assert ps[4, 11] > 0.9

scores = torch.stack([induction_score(get_attention(model, repeated_tokens(50257, 20, tok.bos_token_id, seed=s)), 20) for s in range(5)]).mean(0)
best_ind = top_heads(scores, 6)
assert (5, 5) in best_ind[:3] and scores[5, 5] > 0.8, best_ind
assert all(l >= 5 for l, h in best_ind), "induction heads sit above the previous-token head"
assert scores[:4].max() < 0.2, "no induction heads in the first layers"

print("previous-token heads:", ", ".join(f"L{l}H{h} {ps[l, h]:.2f}" for l, h in best_prev))
print("induction heads (mean over 5 random sequences):", ", ".join(f"L{l}H{h} {scores[l, h]:.2f}" for l, h in best_ind))
print(f"heads with induction score above 0.5: {(scores > 0.5).sum().item()} of 144")

# The effect on the prediction: the second half is predictable, the first is not.
ids = repeated_tokens(50257, 20, tok.bos_token_id, seed=0)
with torch.no_grad():
    lp = torch.log_softmax(model(ids[None]).logits[0], -1)
nll = -lp[:-1].gather(1, ids[1:, None])[:, 0]
print(f"loss on the first half {nll[:20].mean():.2f} nats (random guessing would be {torch.log(torch.tensor(50257.0)):.2f}), on the second half {nll[20:].mean():.2f}")
