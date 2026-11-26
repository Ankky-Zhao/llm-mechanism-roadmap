import math


class Value:
    """A scalar that remembers how it was computed, so it can backpropagate."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                    # dL/d(self), filled in by backward()
        self._backward = lambda: None      # pushes self.grad to the inputs
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # TODO 1: out = self + other. Push out.grad to self.grad and other.grad.
            raise NotImplementedError("TODO 1")

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO 2: out = self * other.
            raise NotImplementedError("TODO 2")

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # TODO 3: out = self ** k.
            raise NotImplementedError("TODO 3")

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # TODO 4: out = tanh(self). Express the local derivative with t.
            raise NotImplementedError("TODO 4")

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            # TODO 5: out = exp(self).
            raise NotImplementedError("TODO 5")

        out._backward = _backward
        return out

    def backward(self):
        # TODO 6: (a) build a topological order of all nodes that self depends on,
        #         (b) set self.grad = 1.0,
        #         (c) call node._backward() from the output back to the leaves.
        raise NotImplementedError("TODO 6")

    # Everything below is built from the operations above. Nothing to change.
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return Value(other) - self
    def __rtruediv__(self, other): return Value(other) / self
