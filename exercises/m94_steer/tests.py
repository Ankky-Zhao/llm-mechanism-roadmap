import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

model, tok, _ = load_models()
EU, AS = tok(" Europe").input_ids[0], tok(" Asia").input_ids[0]
eu, asia = CITIES["Europe"], CITIES["Asia"]
train_eu, test_eu, train_as, test_as = eu[:20], eu[20:], asia[:20], asia[20:]     # the direction comes from one half, the test uses the other

# 1. Positions and states.
ids, pos = prompt(tok, "Paris")
assert tok.decode(ids[0, pos]) == " Paris"
ids, pos = prompt(tok, "Antananarivo")
assert tok.decode(ids[0, :pos + 1]).endswith("Antananarivo") and not tok.decode(ids[0, :pos]).endswith("Antananarivo")
s = city_state(model, tok, "Paris", 6)
with torch.no_grad():
    ref = model(prompt(tok, "Paris")[0], output_hidden_states=True).hidden_states[7][0, 3]
assert s.shape == (768,) and torch.allclose(s, ref, atol=1e-4)

# 2. The direction.
v = mean_difference(model, tok, train_as, train_eu, 6)
assert v.shape == (768,)
assert torch.allclose(v, -mean_difference(model, tok, train_eu, train_as, 6), atol=1e-4)

# 3. No vector and a zero vector change nothing, and the hook is removed.
n_hooks = sum(len(b._forward_hooks) for b in model.transformer.h)
base = steered_logit_diff(model, tok, "Berlin", 6, None, AS, EU)
assert abs(steered_logit_diff(model, tok, "Berlin", 6, torch.zeros(768), AS, EU) - base) < 1e-4
assert sum(len(b._forward_hooks) for b in model.transformer.h) == n_hooks
assert base < 0, "without intervention the model prefers Europe for Berlin"

# 4. Steering held-out European cities toward Asia.
def sweep(layer, vec, alpha):
    d = [steered_logit_diff(model, tok, c, layer, alpha * vec, AS, EU) for c in test_eu]
    return float(np.mean(d)), float(np.mean([x > 0 for x in d]))
base_eu = float(np.mean([steered_logit_diff(model, tok, c, 6, None, AS, EU) for c in test_eu]))
base_as = float(np.mean([steered_logit_diff(model, tok, c, 6, None, AS, EU) for c in test_as]))
g = torch.Generator().manual_seed(0)
rand = torch.randn(768, generator=g); rand = rand / rand.norm() * v.norm()
m2, f2 = sweep(6, v, 2.0)
mr, fr = sweep(6, rand, 2.0)
assert f2 >= 0.8, "twice the mean difference flips most held-out cities"
assert fr <= 0.1 and mr < m2 - 2, "a random direction of the same length does not"

print(f"logit(' Asia') - logit(' Europe') without intervention: held-out European cities {base_eu:+.2f}, held-out Asian cities {base_as:+.2f}")
print("layer  alpha   mean difference   share flipped to Asia   |   random direction of equal norm")
results = {}
for layer in (0, 2, 4, 6, 8, 10):
    vl = mean_difference(model, tok, train_as, train_eu, layer)
    rl = torch.randn(768, generator=g); rl = rl / rl.norm() * vl.norm()
    for alpha in (1.0, 2.0, 4.0):
        m, f = sweep(layer, vl, alpha); mr_, fr_ = sweep(layer, rl, alpha)
        results[(layer, alpha)] = f
        print(f"{layer:>5}  {alpha:>5.0f}   {m:>+15.2f}   {f:>21.2f}   |   {mr_:>+6.2f}  {fr_:.2f}")
assert results[(10, 2.0)] < 0.5, "at layer 10 the city position no longer feeds the answer"

# 5. The weight vector of a logistic-regression probe as steering direction.
X = torch.stack([city_state(model, tok, c, 6) for c in train_eu + train_as]).numpy()
lab = np.array([0] * 20 + [1] * 20)
mu, sd = X.mean(0), X.std(0) + 1e-6
w = torch.tensor(LogisticRegression(C=0.1, max_iter=5000).fit((X - mu) / sd, lab).coef_[0] / sd, dtype=torch.float32)
print(f"layer 6: cosine between the probe weight vector and the mean difference {torch.cosine_similarity(w, v, 0).item():.2f}")
wn = w / w.norm() * v.norm()
for alpha in (1.0, 2.0, 4.0):
    m, f = sweep(6, wn, alpha)
    print(f"  probe direction scaled to the same norm, alpha {alpha:.0f}: mean {m:+.2f}, share flipped {f:.2f}")
