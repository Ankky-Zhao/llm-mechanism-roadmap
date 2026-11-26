import torch


def prev_token_score(att):
    """Mean weight each head puts on the position directly before the reading position.

    att (L, H, T, T) -> (L, H). Position 0 has no previous position and is left out.
    """
    T = att.shape[-1]
    i = torch.arange(1, T)
    return att[:, :, i, i - 1].mean(-1)


def repeated_tokens(vocab_size, n, bos_id, seed=0):
    """BOS followed by n random tokens and the same n tokens again. Returns a LongTensor (2n + 1,)."""
    g = torch.Generator().manual_seed(seed)
    half = torch.randint(0, vocab_size, (n,), generator=g)
    return torch.cat([torch.tensor([bos_id]), half, half])


def induction_score(att, n):
    """Mean weight on the token after the previous occurrence of the current token.

    The sequence is BOS, x_1 .. x_n, x_1 .. x_n. Position i in the second half holds the same token
    as position i - n. An induction head reads from i the position i - n + 1, whose token is the
    one that followed last time. att (L, H, 2n+1, 2n+1) -> (L, H).
    """
    i = torch.arange(n + 1, 2 * n + 1)
    return att[:, :, i, i - n + 1].mean(-1)


def top_heads(score, k):
    """The k heads with the largest score as a list of (layer, head), best first."""
    H = score.shape[1]
    idx = score.flatten().topk(k).indices.tolist()
    return [(j // H, j % H) for j in idx]
