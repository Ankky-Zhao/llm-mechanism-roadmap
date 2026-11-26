import numpy as np
import torch

names, y = city_dataset()
assert len(names) == len(set(names)) == 160 and np.bincount(y).tolist() == [40, 40, 40, 40]

# 1. Synthetic data. A class signal spread over the dimensions is found, pure noise is not.
rng = np.random.default_rng(0)
ys = np.repeat(np.arange(4), 40)
noise = rng.normal(size=(160, 768)).astype(np.float32)
centres = rng.normal(size=(4, 768)).astype(np.float32)
signal = noise + 0.5 * centres[ys]                  # every class is shifted a little in all dimensions
assert probe_cv_accuracy(signal, ys) > 0.9
assert probe_cv_accuracy(noise, ys) < 0.45
# With more dimensions than examples a linear probe fits any labels on its training set.
assert probe_train_accuracy(noise, ys) == 1.0, "768 dimensions, 160 points: every labelling is separable"
assert probe_cv_accuracy(noise, ys, seed=0) == probe_cv_accuracy(noise, ys, seed=0)

# 2. Features from GPT-2.
model, tok, random_model = load_models()
texts = [f"The city of {c}" for c in names]
F = last_token_states(model, tok, texts)
assert F.shape == (160, 13, 768) and F.dtype == np.float32
with torch.no_grad():
    ids = tok(texts[0], return_tensors="pt").input_ids
    ref = model(ids, output_hidden_states=True).hidden_states
assert np.allclose(F[0, 5], ref[5][0, -1].numpy(), atol=1e-4)

# 3. The probe and three controls.
real = [probe_cv_accuracy(F[:, l], y) for l in range(13)]
shuffled_y = np.random.default_rng(0).permutation(y)
shuf = [probe_cv_accuracy(F[:, l], shuffled_y) for l in range(13)]
FR = last_token_states(random_model, tok, texts)
rand = [probe_cv_accuracy(FR[:, l], y) for l in range(13)]
assert max(real) > 0.8, "the continent can be read from the last token of the city name"
assert max(shuf) < 0.4, "shuffled labels cannot be predicted for held-out cities"
assert real[0] > 0.6, "much of it is in the embedding already"
assert max(rand) < min(real), "a model with random weights does worse at every layer"
assert probe_train_accuracy(F[:, 6], shuffled_y) == 1.0

print("held-out accuracy of a linear probe for the continent, chance = 0.25")
print("layer            " + " ".join(f"{l:>5}" for l in range(13)))
print("GPT-2            " + " ".join(f"{a:>5.2f}" for a in real))
print("shuffled labels  " + " ".join(f"{a:>5.2f}" for a in shuf))
print("random weights   " + " ".join(f"{a:>5.2f}" for a in rand))
print(f"training accuracy on shuffled labels at layer 6: {probe_train_accuracy(F[:, 6], shuffled_y):.2f}")

# Why the random model is above chance: the number of tokens of a name differs between the labels.
n_tok = np.array([len(tok(" " + c).input_ids) for c in names])
print("mean number of tokens per city name: " + ", ".join(f"{k} {n_tok[y == i].mean():.2f}" for i, k in enumerate(CONTINENTS)))
print(f"probe on the token count alone: {probe_cv_accuracy(n_tok[:, None].astype(np.float32), y):.2f}")

# Sensitivity to the regularization strength and to the fold assignment.
print("layer 9, C = 0.01, 0.1, 1, 10: " + ", ".join(f"{probe_cv_accuracy(F[:, 9], y, C=c):.2f}" for c in (0.01, 0.1, 1, 10)))
seeds = [probe_cv_accuracy(F[:, 9], y, seed=s) for s in range(5)]
print(f"layer 9, five different fold assignments: {min(seeds):.2f} to {max(seeds):.2f}")

# What the model itself says.
cand = [tok(" " + w).input_ids[0] for w in ("Europe", "Asia", "Africa", "America")]
with torch.no_grad():
    pred = np.array([model(tok(f"The city of {c} is in", return_tensors="pt").input_ids).logits[0, -1][cand].argmax().item() for c in names])
print(f"accuracy of the model's own answer after 'The city of X is in', restricted to four continent tokens: {(pred == y).mean():.2f}"
      + " (" + ", ".join(f"{k} {(pred[y == i] == i).mean():.2f}" for i, k in enumerate(CONTINENTS)) + ")")
