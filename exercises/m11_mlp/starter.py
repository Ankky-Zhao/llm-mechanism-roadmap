import random

# This exercise uses the Value class from the previous exercise.


class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        # TODO 1: return tanh(w . x + b) as a Value. x is a list of floats or Values.
        raise NotImplementedError("TODO 1")

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, sizes):
        dims = [n_in] + sizes
        self.layers = [Layer(dims[i], dims[i + 1]) for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def train(model, xs, ys, steps=300, lr=0.1):
    """Full-batch gradient descent on the mean squared error.

    xs: list of input lists, ys: list of float targets.
    Returns the list of loss values (plain floats), one per step.
    """
    losses = []
    for _ in range(steps):
        # TODO 2: forward pass and mean squared error (a Value)
        # TODO 3: reset every p.grad to 0, then call loss.backward()
        # TODO 4: gradient step  p.data -= lr * p.grad
        # TODO 5: losses.append(loss.data)
        raise NotImplementedError("TODO 2-5")
    return losses
