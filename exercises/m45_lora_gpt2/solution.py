import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class LoRAConv1D(nn.Module):
    """LoRA around one of GPT-2's Conv1D layers:  y = base(x) + (alpha / r) * x @ A @ B"""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        d_in, d_out = base.weight.shape
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(d_in, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, d_out))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A @ self.B)


def add_lora(model, r=8, alpha=16):
    """Freeze every parameter of GPT-2, then wrap the c_attn layer of all 12 blocks. Returns the model."""
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.transformer.h:
        block.attn.c_attn = LoRAConv1D(block.attn.c_attn, r, alpha)
    return model


def count_parameters(model):
    """Returns (trainable, total)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def lora_train(model, tok, pairs, steps=12, lr=2e-3):
    """Like sft_train from module 4.2, but the optimizer only receives the trainable parameters.

    Returns the list of losses.
    """
    ids, attn, labels = collate([build_example(tok, q, a) for q, a in pairs], tok.eos_token_id)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses
