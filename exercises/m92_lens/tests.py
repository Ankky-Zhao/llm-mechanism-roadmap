import torch

model, tok = load_gpt2()
enc = lambda s: tok(s, return_tensors="pt").input_ids[0]

prompt = "The Eiffel Tower is located in the city of"
ids = enc(prompt)
resid = residual_stream(model, ids)

# 1. Thirteen states of size 768, and no hooks left behind.
assert resid.shape == (13, 768), resid.shape
assert len(model.transformer.drop._forward_hooks) == 0 and all(len(b._forward_hooks) == 0 for b in model.transformer.h)

# 2. Row 0 is the embedding sum, the rows agree with the hidden states the model reports.
with torch.no_grad():
    out = model(ids[None], output_hidden_states=True)
    emb = model.transformer.wte(ids[-1]) + model.transformer.wpe(torch.tensor(len(ids) - 1))
assert torch.allclose(resid[0], emb, atol=1e-5)
for l in range(12):
    assert torch.allclose(resid[l], out.hidden_states[l][0, -1], atol=1e-4), l
assert not torch.allclose(resid[12], out.hidden_states[12][0, -1], atol=1e-2), "row 12 is taken before the final LayerNorm"

# 3. The lens applied to the last state is exactly the output of the model.
lens = logit_lens(model, resid)
assert lens.shape == (13, 50257)
assert torch.allclose(lens[-1], out.logits[0, -1], atol=1e-3), "the last row of the lens equals the logits of the model"

# 4. Ranks and KL.
toy = torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]])
assert token_rank(toy, 0) == [3, 1] and token_rank(toy, 2) == [2, 2]
kl = kl_to_final(toy)
p, q = torch.softmax(toy[1], -1), torch.softmax(toy[0], -1)
assert abs(kl[0] - (p * (p / q).log()).sum().item()) < 1e-5 and abs(kl[1]) < 1e-6

paris = enc(" Paris")[0].item()
ranks = token_rank(lens, paris)
kls = kl_to_final(lens)
assert ranks[-1] == 1, "GPT-2 small predicts Paris here"
assert ranks[0] > 1000, "the answer is not readable from the embedding"
assert min(l for l in range(13) if ranks[l] <= 10) >= 6, "the answer enters the top 10 only in the upper half"
assert abs(kls[-1]) < 1e-5 and kls[0] > kls[-2]

print(prompt)
print("layer  top token        rank of ' Paris'   p(' Paris')   KL to final")
for l in range(13):
    top = tok.decode(lens[l].argmax())
    pr = torch.softmax(lens[l], -1)[paris].item()
    print(f"{l:>5}  {top!r:<16} {ranks[l]:>8} {pr:>16.4f} {kls[l]:>12.2f}")

# More prompts: the layer at which the final answer first becomes the top prediction.
prompts = ["The Statue of Liberty is located in the city of New", "Apple was founded by Steve", "Monday, Tuesday, Wednesday,",
           "The currency of Japan is the", "The Brandenburg Gate is located in the city of",
           "When Mary and John went to the store, John gave a drink to", "Barack Obama was born in the state of",
           "The language spoken in Spain is", "The Colosseum is located in the city of"]
print()
print("prompt | final answer | first layer at which it is top | top token at layers 9, 10, 11")
firsts = {}
for p in prompts:
    L = logit_lens(model, residual_stream(model, enc(p)))
    ans = L[-1].argmax().item()
    r = token_rank(L, ans)
    firsts[p] = min(l for l in range(13) if r[l] == 1)
    print(f"{p} | {tok.decode(ans)!r} | {firsts[p]} | " + " ".join(repr(tok.decode(L[l].argmax())) for l in (9, 10, 11)))
assert firsts[prompts[0]] <= 2, "a fixed two-word name is completed almost from the embedding"
assert firsts[prompts[4]] >= 8, "a fact about a named entity appears late"
