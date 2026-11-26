import math
import torch


def adamw_step(p, grad, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    b1, b2 = betas
    p = p - lr * weight_decay * p                       # shrink the weights directly, not through the gradient
    m = b1 * m + (1 - b1) * grad                        # running mean of the gradient
    v = b2 * v + (1 - b2) * grad * grad                 # running mean of the squared gradient
    m_hat = m / (1 - b1 ** t)                           # both start at 0, so early values are too small
    v_hat = v / (1 - b2 ** t)
    p = p - lr * m_hat / (v_hat.sqrt() + eps)
    return p, m, v


def lr_schedule(step, max_lr, warmup_steps, total_steps, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
