import os, tempfile, math
import torch

model, tok = load_gpt2()
text = "The cat sat on the mat because it was tired"
ids = tok(text, return_tensors="pt").input_ids[0]
att = get_attention(model, ids)
T = len(ids)

# 1. Shape: 12 layers, 12 heads, T x T.
assert att.shape == (12, 12, T, T), att.shape

# 2. Every row is a probability distribution.
assert torch.allclose(att.sum(-1), torch.ones(12, 12, T), atol=1e-4), "each row sums to 1"
assert (att >= 0).all()

# 3. The causal mask: no position reads a later one.
upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
assert att[:, :, upper].abs().max() < 1e-6, "weights above the diagonal are 0"
assert torch.allclose(att[:, :, 0, 0], torch.ones(12, 12), atol=1e-5), "the first position can only read itself"

# 4. Entropy. A uniform row over n positions has entropy log n, a one-hot row has entropy 0.
fake = torch.zeros(1, 2, 4, 4)
fake[0, 0] = torch.tril(torch.ones(4, 4)) / torch.arange(1, 5)[:, None]      # uniform over the visible positions
fake[0, 1] = torch.eye(4)                                                    # every position reads itself only
e = attention_entropy(fake)
assert e.shape == (1, 2)
assert abs(e[0, 0].item() - sum(math.log(n) for n in range(1, 5)) / 4) < 1e-5
assert abs(e[0, 1].item()) < 1e-6
ent = attention_entropy(att)
assert ent.shape == (12, 12) and torch.isfinite(ent).all()

# 5. The heatmap is written to disk.
path = os.path.join(tempfile.mkdtemp(), "head.png")
tokens = [tok.decode(i) for i in ids]
plot_head(att, tokens, 4, 11, path)
assert os.path.getsize(path) > 2000

# Observations
sharp = ent.flatten().argmin().item()
flat = ent.flatten().argmax().item()
print(f"tokens: {tokens}")
print(f"sharpest head: layer {sharp // 12} head {sharp % 12}, mean entropy {ent.flatten()[sharp]:.2f} nats")
print(f"most diffuse head: layer {flat // 12} head {flat % 12}, mean entropy {ent.flatten()[flat]:.2f} nats (uniform would be {sum(math.log(n) for n in range(1, T + 1)) / T:.2f})")
print(f"mean weight on the first token over all heads and positions after the first: {att[:, :, 1:, 0].mean():.2f}")
i, c = tokens.index(" it"), tokens.index(" cat")
w = att[:, :, i, c]
for j in w.flatten().topk(3).indices.tolist():
    print(f"layer {j // 12} head {j % 12}, reading from ' it', puts {w.flatten()[j]:.2f} on ' cat'")
print(f"heatmap of layer 4 head 11 saved to {path}")
