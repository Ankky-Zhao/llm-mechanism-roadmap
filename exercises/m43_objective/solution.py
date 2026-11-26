import torch


def kl_regularized_objective(logp, logp_ref, rewards, beta):
    p = logp.exp()
    return (p * rewards).sum() - beta * (p * (logp - logp_ref)).sum()


def optimal_policy(logp_ref, rewards, beta):
    return torch.softmax(logp_ref + rewards / beta, dim=-1)


def ppo_clip_objective(logp_new, logp_old, advantages, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    return torch.min(ratio * advantages, clipped * advantages).mean()


def reinforce_step(theta, logp_ref, rewards, beta, lr=0.5, n_samples=512, generator=None):
    theta = theta.detach().clone().requires_grad_(True)
    logp = torch.log_softmax(theta, dim=-1)
    y = torch.multinomial(logp.exp(), n_samples, replacement=True, generator=generator)
    R = (rewards[y] - beta * (logp[y] - logp_ref[y])).detach()
    loss = -(logp[y] * (R - R.mean())).mean()
    loss.backward()
    return (theta - lr * theta.grad).detach()
