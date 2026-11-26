import math
import torch
import torch.nn.functional as F


def make_batch(data, block_size, batch_size, generator=None):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def lm_loss(logits, targets, ignore_index=-100):
    log_probs = torch.log_softmax(logits, dim=-1)                       # (B, T, V)
    keep = targets != ignore_index                                      # (B, T)
    safe = targets.clamp(min=0)                                         # any valid index where the target is ignored
    picked = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)       # (B, T)
    return -(picked * keep).sum() / keep.sum()


def expected_initial_loss(vocab_size):
    return math.log(vocab_size)


def steps_per_epoch(n_tokens, block_size, batch_size):
    return n_tokens // (block_size * batch_size)
