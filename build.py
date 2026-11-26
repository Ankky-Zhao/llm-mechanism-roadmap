#!/usr/bin/env python3
"""Build index.html from template.html + data/ + exercises/.   Usage: python build.py

No third-party dependencies. The build fails loudly when the data is inconsistent,
so contributors find mistakes before opening a pull request.
"""
import json, pathlib

root = pathlib.Path(__file__).parent
load = lambda p: json.loads((root / p).read_text(encoding="utf-8"))

D = load("data/content.json")
D["resources"] = load("data/resources.json")
D["questions"] = load("data/questions.json")
D["figures"] = {f.name: f.read_text(encoding="utf-8") for f in sorted((root / "data/figures").glob("*.svg"))}

# exercises: one folder each, meta.json plus (for code exercises) starter.py, tests.py, solution.py
D["exercises"] = {}
for d in sorted((root / "exercises").iterdir()):
    if not d.is_dir():
        continue
    ex = load(d / "meta.json")
    if ex["kind"] == "code":
        for name in ("starter", "tests", "solution"):
            ex[name] = (d / f"{name}.py").read_text(encoding="utf-8").rstrip() + "\n"
    D["exercises"][d.name] = ex

# ---- consistency checks ----
M, EX = D["modules"], D["exercises"]
both = lambda o, where: all(isinstance(o.get(k), str) and o[k].strip() for k in ("zh", "en")) or (_ for _ in ()).throw(AssertionError(f"missing zh/en text in {where}"))
used = [m for s in D["stages"] for m in s["modules"]]
assert sorted(used) == sorted(M), f"stage/module mismatch: {set(used) ^ set(M)}"
for k, m in M.items():
    both(m["title"], f"modules.{k}.title")
    if m["status"] != "ready":
        continue
    for field in ("summary", "goals", "explain", "faq", "exercises"):
        assert m.get(field), f"ready module {k} lacks {field}"
    assert isinstance(m.get("derivations"), list), f"ready module {k} lacks derivations (an empty list is allowed where no formula applies)"
    for p in m["explain"]: both(p, f"{k}.explain")
    for f in m["faq"]: both(f["q"], f"{k}.faq.q"); both(f["a"], f"{k}.faq.a")
    for dv in m["derivations"]:
        for st in dv["steps"]:
            assert st["tex"].strip(), f"{k}: empty formula"; both(st["note"], f"{k}.derivation note")
    if m.get("figure"):
        assert m["figure"]["file"] in D["figures"], f"{k}: figure file not found"
    for e in m["exercises"]:
        assert e in EX and EX[e]["module"] == k, f"{k}: exercise {e} missing or assigned to another module"
    assert any(r["topic"] == k for r in D["resources"]), f"{k}: no resources"
    m["notebook"] = f"{m['num']}_{k}.ipynb"
for k, e in EX.items():
    assert e["module"] in M, f"exercise {k}: unknown module"
    both(e["title"], k); both(e["prompt"], k)
    assert e["hints"], f"exercise {k}: at least one hint"
    assert e["kind"] in ("code", "derivation")
    if e["kind"] == "derivation": both(e["solution"], k)
    assert "TODO" in e.get("starter", "TODO"), f"exercise {k}: starter has no TODO"
bad = {r["topic"] for r in D["resources"]} - set(M)
assert not bad, f"resources with unknown topic: {bad}"
urls = [r["url"] for r in D["resources"]]
assert len(urls) == len(set(urls)), "duplicate resource urls"
for r in D["resources"]:
    assert r["level"] in ("intro", "core", "deep") and r["url"].startswith("https://"), r["url"]
    both(r["why"], r["url"])
for s in D["stages"]:
    assert s.get("num"), f"stage {s['id']} needs a num"; both(s["name"], s["id"]); both(s["note"], s["id"])
    for f in ("why", "what", "meaning"): both(s[f], f"{s['id']}.{f}")
GT = {g["term"]["en"] for g in D.get("glossary", [])}
RU = {r["url"] for r in D["resources"]}
for k, m in M.items():
    if m["status"] == "ready":
        both(m["why"], f"{k}.why")
        missing = set(m.get("terms", [])) - GT
        assert not missing, f"{k}: terms not in the glossary: {missing}"
for g in D.get("glossary", []):
    assert g.get("link") is None or g["link"] in RU, f"glossary link for {g['term']['en']} must be a verified entry of resources.json"
    both(g["term"], "glossary"); both(g["def"], g["term"]["en"])
    assert g["module"] is None or g["module"] in M, f"glossary term {g['term']['en']} points to unknown module"
for w in D["whys_planned"]:
    assert w["module"] in M, f"why-question points to unknown module {w['module']}"
for q in D["questions"]:
    assert q["module"] in M and q["region"] in ("cn", "intl") and q["source_type"] in ("report", "collection", "official"), q.get("id")
    assert q.get("sources"), f"question {q.get('id')} has no source link"
    both(q["q"], q.get("id")); both(q["a"], q.get("id"))

tpl = (root / "template.html").read_text(encoding="utf-8")
payload = json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = tpl.replace("__DATA__", payload)
(root / "index.html").write_text(html, encoding="utf-8")
ready = [k for k, m in M.items() if m["status"] == "ready"]
print(f"index.html {len(html)} bytes; {len(ready)}/{len(M)} modules ready; {len(EX)} exercises; {len(D['resources'])} resources; {len(D['questions'])} interview questions; {len(D.get('glossary', []))} glossary terms")
