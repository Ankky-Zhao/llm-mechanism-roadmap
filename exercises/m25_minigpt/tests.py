import codecs, contextlib, io, math
import torch

with contextlib.redirect_stdout(io.StringIO()):
    import this                                   # the Zen of Python, a small text that ships with Python
text = codecs.decode(this.s, "rot13")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
V = len(chars)

torch.manual_seed(0)
model = MiniGPT(V)
x = data[:32][None]
logits, loss = model(x, data[1:33][None])
assert logits.shape == (1, 32, V) and loss.dim() == 0
assert model(x)[1] is None, "no targets, no loss"
assert abs(loss.item() - math.log(V)) < 0.6, f"an untrained model should start near ln(V) = {math.log(V):.2f}, got {loss.item():.2f}"

# causality: changing a later token must not change earlier logits
x2 = x.clone(); x2[0, 20:] = (x2[0, 20:] + 1) % V
assert torch.allclose(model(x)[0][0, :20], model(x2)[0][0, :20], atol=1e-5), "future tokens leak: check the mask"

losses = train(model, data, steps=300)
assert losses[-1] < 1.2, f"final loss {losses[-1]:.2f}: the model should learn this small text"
sample = model.generate(data[:8][None], 120, generator=torch.Generator().manual_seed(0))[0]
out = "".join(chars[i] for i in sample)
print(f"m25_minigpt: all tests passed (loss {losses[0]:.2f} -> {losses[-1]:.2f}, ln V = {math.log(V):.2f})")
print("  sample:", repr(out))
