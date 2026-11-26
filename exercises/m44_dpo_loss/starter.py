import math
import torch
import torch.nn.functional as F


def sequence_logprob(logits, tokens, mask):
    """log pi(y | x) of a whole response: the sum of the log-probabilities of its tokens.

    logits: (B, T, V) where logits[:, t] is the prediction FOR tokens[:, t]  (already aligned)
    tokens: (B, T) long.  mask: (B, T) bool, True on the response tokens that count.
    Returns (B,).
    """
    # TODO 1: log-softmax, pick the token at every position, zero out masked positions, sum over T
    raise NotImplementedError


def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    """Direct Preference Optimization loss, averaged over the batch.

    pi_logp_w / pi_logp_l:   log pi_theta(y_w | x) and log pi_theta(y_l | x), shape (B,)
    ref_logp_w / ref_logp_l: the same for the frozen reference model
    loss = -log sigmoid( beta * [ (pi_w - ref_w) - (pi_l - ref_l) ] )
    """
    # TODO 2
    raise NotImplementedError


def implicit_reward(pi_logp, ref_logp, beta=0.1):
    """The reward that DPO implicitly assigns to a response:  beta * log( pi_theta / pi_ref )."""
    # TODO 3
    raise NotImplementedError
