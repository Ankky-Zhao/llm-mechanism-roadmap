import torch


def kl_regularized_objective(logp, logp_ref, rewards, beta):
    """J(pi) = E_{y ~ pi}[ r(y) ] - beta * KL(pi || pi_ref)   for one prompt with K possible responses.

    logp, logp_ref: (K,) log-probabilities of the policy and of the reference model. rewards: (K,).
    """
    # TODO 1: KL(pi || pi_ref) = sum_y pi(y) * (log pi(y) - log pi_ref(y))
    raise NotImplementedError


def optimal_policy(logp_ref, rewards, beta):
    """The closed-form maximizer of J:  pi*(y) proportional to pi_ref(y) * exp(r(y) / beta).  Returns probabilities (K,)."""
    # TODO 2
    raise NotImplementedError


def ppo_clip_objective(logp_new, logp_old, advantages, eps=0.2):
    """PPO's clipped surrogate objective (to be maximized), averaged over the batch.

    ratio = exp(logp_new - logp_old)
    objective = mean( min( ratio * A,  clip(ratio, 1 - eps, 1 + eps) * A ) )
    """
    # TODO 3
    raise NotImplementedError


def reinforce_step(theta, logp_ref, rewards, beta, lr=0.5, n_samples=512, generator=None):
    """One policy-gradient step on the KL-regularized objective. Returns the new theta (K,).

    The policy is pi = softmax(theta).
      1. sample n_samples responses y from pi           (torch.multinomial, replacement=True)
      2. per-sample return  R = r(y) - beta * (log pi(y) - log pi_ref(y)),  treated as a constant (detach)
      3. subtract the mean of R as a baseline
      4. surrogate loss = -mean( log pi(y) * (R - baseline) );  one gradient-descent step on theta
    """
    # TODO 4
    raise NotImplementedError
