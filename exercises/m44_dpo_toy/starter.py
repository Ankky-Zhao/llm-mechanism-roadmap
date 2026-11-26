import torch
import torch.nn.functional as F

# A toy version of the whole DPO setting. There is one prompt and K possible responses.
# The policy is pi = softmax(theta), the reference is fixed, and annotators compare pairs of
# responses according to a hidden reward (Bradley-Terry model, as in module 4.3).


def sample_preferences(rewards, n_pairs, generator=None):
    """Simulated annotation. Returns (winners, losers), two long tensors of response indices.

    Draw two different responses i, j uniformly; i wins with probability sigmoid(r_i - r_j).
    Nothing to change here.
    """
    K = len(rewards)
    i = torch.randint(0, K, (n_pairs,), generator=generator)
    j = torch.randint(0, K, (n_pairs,), generator=generator)
    keep = i != j
    i, j = i[keep], j[keep]
    i_wins = torch.rand(len(i), generator=generator, dtype=rewards.dtype) < torch.sigmoid(rewards[i] - rewards[j])
    return torch.where(i_wins, i, j), torch.where(i_wins, j, i)


def dpo_train(logp_ref, winners, losers, beta, steps=2000, lr=2.0):
    """Minimise the DPO loss by plain gradient descent, starting from the reference policy.

    theta starts at logp_ref. Each step:
      logp = log_softmax(theta)
      u    = beta * [ (logp[w] - logp_ref[w]) - (logp[l] - logp_ref[l]) ]
      loss = mean( -logsigmoid(u) )
    Returns the final policy as probabilities (K,).
    """
    # TODO 1
    raise NotImplementedError
