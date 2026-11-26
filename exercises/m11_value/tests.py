import torch

def _check(build, inputs, name):
    """Build the same expression with Value and with torch, then compare."""
    vs = [Value(x) for x in inputs]
    out = build(*vs)
    out.backward()
    ts = [torch.tensor(float(x), dtype=torch.float64, requires_grad=True) for x in inputs]
    ref = build(*ts)
    ref.backward()
    assert abs(out.data - ref.item()) < 1e-9, f"{name}: forward {out.data} vs torch {ref.item()}"
    for i, (v, t) in enumerate(zip(vs, ts)):
        assert abs(v.grad - t.grad.item()) < 1e-9, f"{name}: grad of input {i} is {v.grad}, torch says {t.grad.item()}"

# 1. the worked example from the text: L = (a*b + c) * a
_check(lambda a, b, c: (a * b + c) * a, [2.0, -3.0, 10.0], "worked example")
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) * a
L.backward()
assert (a.grad, b.grad, c.grad) == (-2.0, 4.0, 2.0), "worked example should give (-2, 4, 2)"

# 2. a node used several times: gradients must accumulate
_check(lambda a: a + a, [3.0], "a + a")
_check(lambda a, b: (a + b) * (a + b) * a, [1.5, -0.5], "reused node")

# 3. one tanh neuron
_check(lambda x1, x2, w1, w2, b: (x1 * w1 + x2 * w2 + b).tanh(), [2.0, 0.0, -3.0, 1.0, 6.88], "tanh neuron")

# 4. powers, division, subtraction, exp
_check(lambda a, b: (a ** 3 - b) / (a * b + 2.0), [1.2, 0.7], "pow and div")
_check(lambda a, b: (a * b).exp() + (2.0 * a - b).exp(), [0.3, -0.4], "exp")

# 5. a deeper chain, to make sure the order of the backward pass is right
def _deep(a, b):
    x = a
    for _ in range(5):
        x = (x * b + a).tanh() * x + b
    return x
_check(_deep, [0.4, -0.9], "deep chain")
print("m11_value: all tests passed")
