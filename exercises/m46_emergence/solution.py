import math


def per_token_accuracy(n_params, slope=0.1165, at_1e6=0.30):
    return min(0.999, at_1e6 + slope * (math.log10(n_params) - 6))


def exact_match(p, length):
    return p ** length


def biggest_jump(values):
    return max(b - a for a, b in zip(values, values[1:]))


def threshold_scale(sizes, scores, level=0.5):
    for n, s in zip(sizes, scores):
        if s >= level:
            return n
    return None
