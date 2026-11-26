import io
import sentencepiece as spm


def train_sentencepiece(lines, vocab_size, model_type="bpe"):
    """Train a SentencePiece model in memory and return a ready SentencePieceProcessor.

    lines: list of strings (raw text, no pre-tokenization needed).  model_type: "bpe" or "unigram".
    """
    model = io.BytesIO()
    # TODO 1: spm.SentencePieceTrainer.train(sentence_iterator=iter(lines), model_writer=model,
    #                                         vocab_size=..., model_type=..., minloglevel=2)
    # TODO 2: return spm.SentencePieceProcessor(model_proto=model.getvalue())
    raise NotImplementedError


def detokenize(pieces):
    """Turn SentencePiece pieces back into text WITHOUT the library.

    SentencePiece writes a space as the visible symbol '▁' (U+2581) at the start of a piece.
    """
    # TODO 3: join the pieces, turn '▁' back into a space, drop the space at the very beginning
    raise NotImplementedError
