import torch
import torch.nn.functional as F


def preference_prob(r_a, r_b):
    """Bradley-Terry model: the probability that a is preferred over b, given their reward scores."""
    # TODO 1: sigmoid of the difference
    raise NotImplementedError


def pairwise_loss(r_winner, r_loser):
    """Reward-model loss for a batch of comparisons: mean of  -log sigmoid(r_winner - r_loser).

    Use F.logsigmoid for numerical stability.
    """
    # TODO 2
    raise NotImplementedError


def train_reward_model(features, winners, losers, steps=300, lr=0.1):
    """Fit a linear reward model r(x) = x @ w from comparisons.

    features: (N, d) float tensor, one row per response.
    winners, losers: long tensors (M,). In comparison j, response winners[j] was preferred over losers[j].
    Start from w = 0, run plain gradient descent on pairwise_loss, return w as a (d,) tensor.
    """
    # TODO 3
    raise NotImplementedError
