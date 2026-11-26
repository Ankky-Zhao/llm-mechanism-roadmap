import re

def chunk_words(text, size, overlap=0):
    """Windows of `size` words that share `overlap` words with their neighbour. Returns a list of strings."""
    assert 0 <= overlap < size
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks

def chunk_sentences(text, max_words):
    """Pack whole sentences into chunks of at most `max_words` words. Returns a list of strings."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    chunks, current, count = [], [], 0
    for s in sentences:
        n = len(s.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current)); current, count = [], 0
        current.append(s); count += n
    if current:
        chunks.append(" ".join(current))
    return chunks
