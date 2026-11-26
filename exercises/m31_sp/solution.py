import io
import sentencepiece as spm


def train_sentencepiece(lines, vocab_size, model_type="bpe"):
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(lines), model_writer=model,
                                   vocab_size=vocab_size, model_type=model_type, minloglevel=2)
    return spm.SentencePieceProcessor(model_proto=model.getvalue())


def detokenize(pieces):
    text = "".join(pieces).replace("▁", " ")
    return text[1:] if text.startswith(" ") else text
