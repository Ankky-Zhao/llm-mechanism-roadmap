import torch
import torch.nn.functional as F


def preference_prob(r_a, r_b):
    return torch.sigmoid(r_a - r_b)


def pairwise_loss(r_winner, r_loser):
    return -F.logsigmoid(r_winner - r_loser).mean()


def train_reward_model(features, winners, losers, steps=300, lr=0.1):
    w = torch.zeros(features.shape[1], dtype=features.dtype, requires_grad=True)
    for _ in range(steps):
        r = features @ w                                   # one score per response
        loss = pairwise_loss(r[winners], r[losers])
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            w.grad.zero_()
    return w.detach()
