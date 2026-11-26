import math
import re
import string
from collections import Counter


def normalize(text):
    """The normalisation used for question-answering metrics: lower case, remove punctuation
    (string.punctuation), remove the articles a / an / the as whole words, collapse whitespace."""
    # TODO 1
    raise NotImplementedError


def exact_match(prediction, reference):
    """1.0 if the two strings are equal after normalize, else 0.0."""
    # TODO 2
    raise NotImplementedError


def token_f1(prediction, reference):
    """F1 over the normalised tokens. precision = common / len(prediction), recall = common / len(reference),
    where `common` counts shared tokens with multiplicity (Counter(p) & Counter(r)).
    If either side is empty, return 1.0 when both are empty and 0.0 otherwise."""
    # TODO 3
    raise NotImplementedError


def ngrams(tokens, n):
    """A Counter of the n-grams (as tuples) of a token list. Nothing to change here."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate, reference, max_n=4):
    """Sentence-level BLEU with one reference and no smoothing. Tokens: lower-cased, split on whitespace.

    For n = 1..max_n the modified precision p_n = (clipped matches) / (n-grams in the candidate),
    where an n-gram of the candidate counts at most as often as it occurs in the reference.
    BLEU = BP * exp( mean of log p_n ),   BP = 1 if len(c) > len(r) else exp(1 - len(r) / len(c)).
    If any p_n is 0 (or the candidate has no n-grams of that length) the result is 0.0.
    """
    # TODO 4
    raise NotImplementedError


def lcs_length(a, b):
    """Length of the longest common subsequence of two token lists (dynamic programming)."""
    # TODO 5
    raise NotImplementedError


def rouge_l(candidate, reference):
    """ROUGE-L as an F1: precision = LCS / len(candidate), recall = LCS / len(reference). 0.0 if LCS is 0."""
    # TODO 6
    raise NotImplementedError
