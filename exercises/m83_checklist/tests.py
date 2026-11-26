from huggingface_hub import ModelCard

FULL = """---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen2.5-0.5B-Instruct
---
# GeoQA-mini

A small model for questions about an invented town.

## Intended uses
Teaching material. Not for decisions about people.

## Limitations and bias
Answers may be wrong; see the evaluation.

## Training data
30 hand-written passages.

### Evaluation results
16 of 20 questions correct with retrieval.
"""
THIN = """---
tags:
- text-generation
---
# my-model

## Quickstart
Run it.
"""

meta, body = parse_card(FULL)
assert meta["license"] == "apache-2.0" and meta["language"] == ["en"] and body.lstrip().startswith("# GeoQA-mini")
assert headings(body) == ["geoqa-mini", "intended uses", "limitations and bias", "training data", "evaluation results"]
full, thin = check_card(FULL), check_card(THIN)
assert full == {"license": True, "language": True, "intended_use": True, "limitations": True, "training_data": True, "evaluation": True}
assert thin == {"license": False, "language": False, "intended_use": False, "limitations": False, "training_data": False, "evaluation": False}
assert completeness(full) == 1.0 and completeness(thin) == 0.0
half = check_card("---\nlicense: mit\n---\n# x\n\n## Evaluation\nok\n\n## Known risks\nsome\n")
assert half["license"] and half["evaluation"] and half["limitations"] and not half["language"] and completeness(half) == 0.5
assert check_card("# no metadata at all\n\n## Training\nx\n")["training_data"] is True

# Real cards from the Hub. Only the old GPT-2 card is asserted; the others are printed, since cards get edited.
gpt2 = check_card(ModelCard.load("openai-community/gpt2").content)
assert gpt2["license"] and gpt2["limitations"] and gpt2["training_data"] and gpt2["evaluation"]
rows = [("openai-community/gpt2", gpt2)]
for repo in ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B-Instruct", "sentence-transformers/all-MiniLM-L6-v2", "cross-encoder/nli-deberta-v3-small"):
    try:
        rows.append((repo, check_card(ModelCard.load(repo).content)))
    except Exception as e:
        rows.append((repo, None))

print("m83_checklist: all tests passed")
print(f"  {'model card':42s} license language intended limits training eval   score")
for repo, r in rows:
    if r is None:
        print(f"  {repo:42s} (could not be loaded)")
        continue
    mark = lambda b: "yes" if b else " - "
    print(f"  {repo:42s} {mark(r['license']):7s} {mark(r['language']):8s} {mark(r['intended_use']):8s} {mark(r['limitations']):6s} {mark(r['training_data']):8s} {mark(r['evaluation']):5s} {completeness(r):5.0%}")
