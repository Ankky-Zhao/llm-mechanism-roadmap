import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# Everything from exercises 1 to 3 is defined above: BM25, embed, dense_scores, ranking, rrf, rerank.

RAG_INSTRUCTION = ("Use only the following passages to answer the question. "
                   "If the answer is not in the passages, say \"I don't know\".")


def build_rag_prompt(question, passages):
    """The prompt that carries the retrieved passages. Exactly this layout:

    <RAG_INSTRUCTION>
    <blank line>
    [1] first passage
    [2] second passage
    <blank line>
    Question: <question>
    Answer in one short sentence.
    """
    # TODO 1
    raise NotImplementedError


@torch.no_grad()
def chat(model, tokenizer, user_message, max_new_tokens=40):
    """One user message in, the reply of the chat model out (greedy decoding, special tokens removed).

    enc = tokenizer.apply_chat_template([{"role": "user", "content": ...}], add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)
    Decode only the tokens generated after the prompt.
    """
    # TODO 2
    raise NotImplementedError


def retrieve(question, corpus, bm25, doc_vectors, embedder, embed_tok, cross, cross_tok, k=3):
    """The retrieval pipeline of exercises 1 to 3 in one function. Returns the indices of the k best passages.

    1. rank by embeddings   2. rank by BM25   3. fuse the two with rrf   4. rerank the fused list   5. keep k
    """
    # TODO 3
    raise NotImplementedError


def rag_answer(question, corpus, retriever, model, tokenizer):
    """Retrieve, build the prompt, generate. `retriever` is a function question -> list of passage indices.
    Returns (reply, ids)."""
    # TODO 4
    raise NotImplementedError
