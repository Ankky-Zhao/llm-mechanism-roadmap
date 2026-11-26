import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
N = 12
show = lambda ids, T: repr(tok.decode(ids[T:]))

rows = []
for prompt in ["The capital of France is", "My favourite food is", "In machine learning, attention is"]:
    ids = tok(prompt, return_tensors="pt").input_ids
    T = ids.shape[1]

    # 1. Beam size 1 is greedy decoding.
    greedy = model.generate(ids, do_sample=False, max_new_tokens=N, pad_token_id=tok.eos_token_id)
    b1, s1 = beam_search(model, ids, 1, N)
    assert b1.shape == (1, T + N) and torch.equal(b1[0], greedy[0]), "beam size 1 must reproduce greedy decoding"

    # 2. The returned score is the log-probability of the continuation.
    assert abs(s1[0].item() - sequence_logprob(model, b1[:1], T)) < 1e-2

    # 3. Beam size 4 agrees with the Hugging Face implementation and is sorted best first.
    b4, s4 = beam_search(model, ids, 4, N)
    hf = model.generate(ids, num_beams=4, do_sample=False, max_new_tokens=N, pad_token_id=tok.eos_token_id)
    assert b4.shape == (4, T + N) and torch.equal(b4[0], hf[0]), "beam size 4 should match model.generate(num_beams=4)"
    assert (s4[:-1] >= s4[1:]).all()
    assert abs(s4[0].item() - sequence_logprob(model, b4[:1], T)) < 1e-2

    # 4. A wider beam never finds a less probable sequence than greedy here.
    assert s4[0] >= s1[0] - 1e-4
    rows.append((prompt, s1[0].item(), show(b1[0], T), s4[0].item(), show(b4[0], T)))

assert any(r[3] > r[1] + 0.5 for r in rows), "on these prompts beam search finds clearly more probable continuations"
print("m51_beam: all tests passed")
for prompt, sg, tg, sb, tb in rows:
    print(f"  {prompt!r}")
    print(f"    greedy  log p = {sg:7.2f}  {tg}")
    print(f"    beam 4  log p = {sb:7.2f}  {tb}")
