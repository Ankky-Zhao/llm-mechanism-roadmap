import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
torch.manual_seed(0)

ids, labels = build_example(tok, "What is the capital of France?", "Paris")
assert len(ids) == len(labels)
n_prompt = len(tok(format_prompt("What is the capital of France?")).input_ids)
assert labels[:n_prompt] == [-100] * n_prompt, "no loss on the prompt"
assert labels[n_prompt:] == ids[n_prompt:], "the answer tokens are their own targets"
assert tok.decode(ids[n_prompt:]) == "ANSWER: Paris.<|endoftext|>", "the end token is part of what the model must learn"

pairs = [("What is the capital of France?", "Paris"), ("What is the capital of Germany?", "Berlin"),
         ("What is the capital of Italy?", "Rome"), ("What is the capital of Spain?", "Madrid"),
         ("What is the capital of Japan?", "Tokyo"), ("What is the capital of China?", "Beijing"),
         ("What is the capital of Russia?", "Moscow"), ("What is the capital of Egypt?", "Cairo")]
held_out = "What is the capital of Poland?"
general = "The weather was cold and the streets were quiet as people walked home from work in the evening."

before = answer(model, tok, held_out)
ppl_before = perplexity(model, tok, general)
assert not before.startswith("ANSWER:"), "the pretrained model does not know the format yet"

losses = sft_train(model, tok, pairs, steps=12, lr=1e-4)
assert len(losses) == 12 and losses[-1] < 0.1 * losses[0], f"the loss should drop sharply: {losses[0]:.2f} -> {losses[-1]:.4f}"

after = answer(model, tok, held_out)
ppl_after = perplexity(model, tok, general)
assert after.startswith("ANSWER:"), f"after fine-tuning the model should answer in the format, got {after!r}"
assert ppl_after > ppl_before, "fine-tuning on narrow data makes the model worse on ordinary text (forgetting)"
print("m42_sft: all tests passed")
print(f"  before: {before!r}")
print(f"  after : {after!r}     (this question was not in the training set)")
print(f"  perplexity on ordinary text: {ppl_before:.1f} -> {ppl_after:.1f}")
