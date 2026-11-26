import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

RAG_INSTRUCTION = ("Use only the following passages to answer the question. "
                   "If the answer is not in the passages, say \"I don't know\".")


def build_rag_prompt(question, passages):
    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"{RAG_INSTRUCTION}\n\n{numbered}\n\nQuestion: {question}\nAnswer in one short sentence."


@torch.no_grad()
def chat(model, tokenizer, user_message, max_new_tokens=40):
    enc = tokenizer.apply_chat_template([{"role": "user", "content": user_message}], add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def retrieve(question, corpus, bm25, doc_vectors, embedder, embed_tok, cross, cross_tok, k=3):
    first = ranking(dense_scores(doc_vectors, embed(embedder, embed_tok, [question])[0]))
    fused = rrf([ranking(bm25.scores(question)), first])
    return rerank(cross, cross_tok, question, corpus, fused)[:k]


def rag_answer(question, corpus, retriever, model, tokenizer):
    ids = retriever(question)
    reply = chat(model, tokenizer, build_rag_prompt(question, [corpus[i] for i in ids]))
    return reply, ids
