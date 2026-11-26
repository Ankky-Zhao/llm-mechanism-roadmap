import re

def chunk_words(text, size, overlap=0):
    """Windows of `size` words that share `overlap` words with their neighbour. Returns a list of strings."""
    assert 0 <= overlap < size
    words = text.split()
    # TODO: slide a window of `size` words forward by size - overlap, stop once the end is covered
    raise NotImplementedError

def chunk_sentences(text, max_words):
    """Pack whole sentences into chunks of at most `max_words` words. Returns a list of strings."""
    # TODO: split into sentences, then add sentences to the current chunk while they fit
    raise NotImplementedError
