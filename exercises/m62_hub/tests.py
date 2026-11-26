# 1. Classification on fixed inputs
cases = [("apache-2.0", None, "permissive"), ("MIT", None, "permissive"), ("cc-by-4.0", None, "permissive"),
         ("gpl-3.0", None, "copyleft"), ("cc-by-nc-4.0", None, "non-commercial"), ("cc-by-nc-sa-4.0", None, "non-commercial"),
         ("fair-noncommercial-research-license", None, "non-commercial"), ("other", "mrl", "non-commercial"),
         ("other", "some-research-license", "non-commercial"),
         ("openrail", None, "use-restricted"), ("bigscience-bloom-rail-1.0", None, "use-restricted"),
         ("llama3.3", None, "custom"), ("gemma", None, "custom"), ("other", "llama4", "custom"), ("other", None, "custom"),
         (None, None, "unknown"), ("", None, "unknown"), ("unknown", None, "unknown")]
for ident, name, expected in cases:
    got = classify_license(ident, name)
    assert got == expected, f"classify_license({ident!r}, {name!r}) should be {expected!r}, got {got!r}"

# 2. Live metadata from the Hub. These repositories are old and their licenses are unlikely to change.
row = read_license("openai-community/gpt2")
assert row == {"repo": "openai-community/gpt2", "license": "mit", "name": None, "gated": False, "category": "permissive"}, row
row = read_license("meta-llama/Llama-3.3-70B-Instruct")       # metadata of gated repositories is public
assert row["license"] == "llama3.3" and row["category"] == "custom" and row["gated"] is True, row
row = read_license("mistralai/Mistral-Large-Instruct-2411")
assert row["license"] == "other" and row["name"] == "mrl" and row["category"] == "non-commercial", row

# 3. A survey of well-known checkpoints. Printed, not asserted: this is exactly the information that changes.
REPOS = ["openai-community/gpt2", "HuggingFaceTB/SmolLM2-135M", "openai/gpt-oss-20b", "microsoft/phi-4",
         "deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V3", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-2.4T-A95B",
         "google/gemma-3-27b-it", "google/gemma-4-12B-it", "meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
         "zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash", "moonshotai/Kimi-K3.1-Pro", "mistralai/Mistral-Large-Instruct-2411",
         "mistralai/Mistral-Large-4-1T-A52B", "CohereLabs/c4ai-command-a-03-2025", "bigscience/bloom"]
rows = []
for r in REPOS:
    try:
        rows.append(read_license(r))
    except Exception as e:                                     # a repository may be renamed or removed later
        rows.append({"repo": r, "license": f"<{type(e).__name__}>", "name": None, "gated": False, "category": "unknown"})
assert len(survey(REPOS[:2])) == 2 and survey(REPOS[:2])[1]["category"] == "permissive"

print("m62_hub: all tests passed")
print(f"  {'repository':44s} {'license':26s} {'gated':6s} category")
for x in rows:
    lic = str(x["license"]) + (f" ({x['name']})" if x["name"] else "")
    print(f"  {x['repo']:44s} {lic:26s} {'yes' if x['gated'] else 'no':6s} {x['category']}")
