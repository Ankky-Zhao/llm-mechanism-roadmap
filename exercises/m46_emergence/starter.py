import math


def per_token_accuracy(n_params, slope=0.1165, at_1e6=0.30):
    """A made-up but smooth law: accuracy on a single token grows linearly in log10(parameters).

    Capped at 0.999. Nothing to change here.
    """
    return min(0.999, at_1e6 + slope * (math.log10(n_params) - 6))


def exact_match(p, length):
    """Probability of getting ALL `length` tokens of an answer right, if each is right with probability p (independently)."""
    # TODO 1
    raise NotImplementedError


def biggest_jump(values):
    """The largest increase between two neighbouring entries of a list."""
    # TODO 2
    raise NotImplementedError


def threshold_scale(sizes, scores, level=0.5):
    """The first model size whose score reaches `level`, or None if none does."""
    # TODO 3
    raise NotImplementedError
