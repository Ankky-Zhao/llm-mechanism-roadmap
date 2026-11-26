import random

random.seed(0)
xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [-1.0, 1.0, 1.0, -1.0]                     # XOR with targets in {-1, +1}

model = MLP(2, [4, 1])
assert len(model.parameters()) == 4 * 3 + 5, "a 2-4-1 MLP has 17 parameters"

n = Neuron(2)
n.w[0].data, n.w[1].data, n.b.data = 0.5, -1.0, 0.25
out = n([2.0, 1.0])
import math
assert abs(out.data - math.tanh(0.25)) < 1e-12, "Neuron should return tanh(w . x + b)"

losses = train(model, xs, ys, steps=300, lr=0.1)
assert len(losses) == 300 and all(isinstance(l, float) for l in losses), "return one float per step"
assert losses[-1] < losses[0], "the loss should go down"
assert losses[-1] < 0.05, f"final loss {losses[-1]:.4f} is too high. Did you reset the gradients every step?"
for x, y in zip(xs, ys):
    assert model(x).data * y > 0, f"wrong sign for input {x}"
print(f"m11_mlp: all tests passed (loss {losses[0]:.3f} -> {losses[-1]:.4f})")
