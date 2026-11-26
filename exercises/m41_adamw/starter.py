import math
import torch


def adamw_step(p, grad, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """One AdamW update, without torch.optim.

    p, grad, m, v: tensors of the same shape (parameter, its gradient, first and second moment).
    t: the step number, starting at 1.
    Returns the new (p, m, v). Do not modify the inputs in place.
      p <- p - lr * weight_decay * p                     (decoupled weight decay)
      m <- b1 * m + (1 - b1) * grad
      v <- b2 * v + (1 - b2) * grad^2
      m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)    (bias correction)
      p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    """
    # TODO 1
    raise NotImplementedError


def lr_schedule(step, max_lr, warmup_steps, total_steps, min_lr=0.0):
    """Linear warm-up followed by cosine decay.

    step 0 .. warmup_steps-1:  max_lr * (step + 1) / warmup_steps
    afterwards:                min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
                               with progress = (step - warmup_steps) / (total_steps - warmup_steps)
    at or beyond total_steps:  min_lr
    """
    # TODO 2
    raise NotImplementedError
