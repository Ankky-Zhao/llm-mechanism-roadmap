import torch
import torch.nn.functional as F


def sample_preferences(rewards, n_pairs, generator=None):
    K = len(rewards)
    i = torch.randint(0, K, (n_pairs,), generator=generator)
    j = torch.randint(0, K, (n_pairs,), generator=generator)
    keep = i != j
    i, j = i[keep], j[keep]
    i_wins = torch.rand(len(i), generator=generator, dtype=rewards.dtype) < torch.sigmoid(rewards[i] - rewards[j])
    return torch.where(i_wins, i, j), torch.where(i_wins, j, i)


def dpo_train(logp_ref, winners, losers, beta, steps=2000, lr=2.0):
    theta = logp_ref.clone().requires_grad_(True)
    for _ in range(steps):
        logp = torch.log_softmax(theta, dim=-1)
        u = beta * ((logp[winners] - logp_ref[winners]) - (logp[losers] - logp_ref[losers]))
        loss = -F.logsigmoid(u).mean()
        loss.backward()
        with torch.no_grad():
            theta -= lr * theta.grad
            theta.grad.zero_()
    return torch.softmax(theta.detach(), dim=-1)
