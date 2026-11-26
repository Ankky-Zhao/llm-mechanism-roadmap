def pair_counts(ids):
    """Count every adjacent pair in the list `ids`.

    Returns a dict {(a, b): count}. Keys must be inserted in order of first appearance
    (a plain dict filled by one left-to-right pass does this), so that ties are broken
    the same way for everyone.
    """
    # TODO 1
    raise NotImplementedError


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids`, left to right, by `new_id`."""
    # TODO 2: walk through ids with an index; skip two items after a match
    raise NotImplementedError


def train_bpe(text, vocab_size):
    """Byte-level BPE training.

    Start from the UTF-8 bytes of `text` (ids 0..255). Repeatedly take the most frequent
    pair (first one on ties: max(counts, key=counts.get)), give it the next free id
    (256, 257, ...) and merge it, until the vocabulary has `vocab_size` entries or no pair is left.
    Returns the dict {(a, b): new_id} in the order the merges were learned.
    """
    # TODO 3
    raise NotImplementedError


def encode(text, merges):
    """Text -> list of token ids, applying the learned merges in the order they were learned."""
    ids = list(text.encode("utf-8"))
    # TODO 4: while a mergeable pair exists, merge the one that was learned EARLIEST
    #         (the pair with the smallest merges[pair])
    raise NotImplementedError


def decode(ids, merges):
    """List of token ids -> text."""
    # TODO 5: build {id: bytes} for 0..255 and for every merge, concatenate, decode as UTF-8
    #         (use errors="replace" so that a cut-off character does not crash)
    raise NotImplementedError
