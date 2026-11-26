def pair_counts(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merges = {}
    for new_id in range(256, vocab_size):
        counts = pair_counts(ids)
        if not counts:
            break
        pair = max(counts, key=counts.get)        # most frequent pair, first one on ties
        merges[pair] = new_id
        ids = merge(ids, pair, new_id)
    return merges


def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        counts = pair_counts(ids)
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))   # earliest learned merge present
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():          # dicts keep insertion order, so a and b already exist
        vocab[new_id] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
