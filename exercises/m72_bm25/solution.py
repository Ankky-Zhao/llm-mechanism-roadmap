import math
import re
from collections import Counter
import torch
from transformers import AutoModel, AutoTokenizer


def tokenize(text):
    """Lower-case words and numbers; identifiers such as lh-204 stay in one piece. Nothing to change here."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(term for d in self.docs for term in set(d))
        self.idf = {term: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for term, n in df.items()}

    def scores(self, query):
        out = []
        for doc, tf in zip(self.docs, self.tf):
            norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
            s = 0.0
            for term in tokenize(query):
                f = tf.get(term, 0)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


@torch.no_grad()
def embed(model, tokenizer, texts):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    hidden = model(**enc).last_hidden_state                       # (n, T, d)
    mask = enc["attention_mask"][..., None].float()               # padding positions must not count
    mean = (hidden * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(mean, dim=-1)


def dense_scores(doc_vectors, query_vector):
    return (doc_vectors @ query_vector).tolist()                  # cosine similarity, since all vectors have length 1
