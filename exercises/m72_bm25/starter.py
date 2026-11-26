import math
import re
from collections import Counter
import torch
from transformers import AutoModel, AutoTokenizer


def tokenize(text):
    """Lower-case words and numbers; identifiers such as lh-204 stay in one piece. Nothing to change here."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class BM25:
    """Okapi BM25, the standard keyword ranking function.

    score(q, d) = sum over the query terms t of
        idf(t) * f(t, d) * (k1 + 1) / ( f(t, d) + k1 * (1 - b + b * |d| / avgdl) )
    idf(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )

    f(t, d): how often t occurs in d      |d|: number of tokens in d      avgdl: mean document length
    N: number of documents                n_t: number of documents that contain t
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        # TODO 1: tokenise every document; store self.docs, self.N, self.avgdl,
        #         self.tf (a Counter per document) and self.idf (a dict term -> idf)
        raise NotImplementedError

    def scores(self, query):
        """The BM25 score of every document for this query, as a list of floats.
        A query term that occurs twice in the query counts twice. Terms not in a document add nothing."""
        # TODO 2
        raise NotImplementedError


@torch.no_grad()
def embed(model, tokenizer, texts):
    """Sentence embeddings by mean pooling, as a tensor (n, d) with rows of length 1.

    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    Average model(**enc).last_hidden_state over the REAL tokens of each text (use enc["attention_mask"],
    padding must not count), then normalise every row to unit length.
    """
    # TODO 3
    raise NotImplementedError


def dense_scores(doc_vectors, query_vector):
    """Cosine similarity of every document vector (n, d) with one query vector (d,), as a list of floats."""
    # TODO 4
    raise NotImplementedError
