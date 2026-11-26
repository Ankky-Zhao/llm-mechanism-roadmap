import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# This exercise uses your MultiHeadAttention class from module 2.2.


class Block(nn.Module):
    """A Pre-Norm Transformer block."""

    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, mask):
        # TODO 1: x = x + attention(ln1(x)) with the causal mask   (self.attn returns (output, weights))
        # TODO 2: x = x + mlp(ln2(x))
        raise NotImplementedError


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size=32, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns (logits (B, T, V), loss or None)."""
        B, T = idx.shape
        # TODO 3: token embedding + position embedding
        # TODO 4: build the causal mask (T, T), True = may attend, and run the blocks
        # TODO 5: final LayerNorm, then the head -> logits
        # TODO 6: if targets is given, loss = cross-entropy between logits (B*T, V) and targets (B*T,)
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, generator=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = torch.softmax(logits[:, -1], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1, generator=generator)], dim=1)
        return idx


def train(model, data, steps=300, batch_size=32, lr=3e-3, seed=0):
    """Train on a 1-D tensor of token ids. Returns the list of loss values. Nothing to change here."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        ix = torch.randint(len(data) - model.block_size - 1, (batch_size,), generator=g)
        x = torch.stack([data[i:i + model.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + model.block_size + 1] for i in ix])     # the same text, shifted by one
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
