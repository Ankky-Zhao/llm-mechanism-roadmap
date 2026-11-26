import math
import torch
import torch.nn.functional as F


def sequence_logprob(logits, tokens, mask):
    logp = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)   # (B, T)
    return (logp * mask).sum(dim=-1)


def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    u = beta * ((pi_logp_w - ref_logp_w) - (pi_logp_l - ref_logp_l))
    return -F.logsigmoid(u).mean()


def implicit_reward(pi_logp, ref_logp, beta=0.1):
    return beta * (pi_logp - ref_logp)
