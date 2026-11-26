import math
import torch
import torch.nn.functional as F


def make_batch(data, block_size, batch_size, generator=None):
    """Cut training examples out of one long token sequence.

    data: 1-D long tensor of token ids.
    Pick batch_size random start positions i with torch.randint(len(data) - block_size, (batch_size,), generator=generator).
    Returns (x, y), both (batch_size, block_size):
      x[b] = data[i : i + block_size]
      y[b] = data[i + 1 : i + block_size + 1]      the same text shifted by one token
    """
    # TODO 1
    raise NotImplementedError


def lm_loss(logits, targets, ignore_index=-100):
    """Next-token loss. logits: (B, T, V), targets: (B, T).

    Mean negative log-likelihood over all positions whose target is not ignore_index.
    Write it without F.cross_entropy: log-softmax, pick the target, mask, average.
    """
    # TODO 2
    raise NotImplementedError


def expected_initial_loss(vocab_size):
    """The loss of a model that predicts every token with equal probability."""
    # TODO 3
    raise NotImplementedError


def steps_per_epoch(n_tokens, block_size, batch_size):
    """How many optimizer steps it takes to see n_tokens tokens once (round down)."""
    # TODO 4
    raise NotImplementedError
