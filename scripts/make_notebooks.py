#!/usr/bin/env python3
"""Generate one notebook per finished module into notebooks/.   Usage: python scripts/make_notebooks.py

Notebooks are build products: edit exercises/<id>/ and data/content.json, then regenerate.
Standard library only.
"""
import json, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
C = json.loads((root / "data/content.json").read_text(encoding="utf-8"))
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
fold = lambda title, body: f"<details><summary><b>{title}</b></summary>\n\n{body}\n\n</details>"

(root / "notebooks").mkdir(exist_ok=True)
for k, m in C["modules"].items():
    if m["status"] != "ready":
        continue
    cells = [md(f"# {m['num']} {m['title']['zh']} / {m['title']['en']}\n\n{m['summary']['zh']}\n\n{m['summary']['en']}\n\n"
                f"讲解与推导见网站对应模块。每个练习先写代码，再运行紧随其后的检查单元。\n\n"
                f"The explanation and derivations are on the website. For each exercise, write the code, then run the check cell right after it.")]
    for i, e in enumerate(m["exercises"], 1):
        d = root / "exercises" / e
        x = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        head = f"## {i}. {x['title']['zh']} / {x['title']['en']}\n\n{x['prompt']['zh']}\n\n{x['prompt']['en']}"
        if x.get("network"):
            head += "\n\n需要联网下载模型。/ Downloads a model."
        cells.append(md(head))
        hints = "\n\n".join(fold(f"提示 {j} / Hint {j}", h["zh"] + "\n\n" + h["en"]) for j, h in enumerate(x["hints"], 1))
        if x["kind"] == "code":
            for dep in x.get("requires", []):
                if dep not in m["exercises"]:       # code from another module that this exercise builds on
                    cells.append(code(f"# 来自练习 {dep} 的参考实现，本题会用到 / reference code from exercise {dep}, used below\n"
                                      + (root / "exercises" / dep / "solution.py").read_text(encoding="utf-8").rstrip()))
            cells.append(code((d / "starter.py").read_text(encoding="utf-8").rstrip()))
            cells.append(code("# 检查 / checks\n" + (d / "tests.py").read_text(encoding="utf-8").rstrip()))
            sol = "```python\n" + (d / "solution.py").read_text(encoding="utf-8").rstrip() + "\n```"
        else:
            sol = x["solution"]["zh"] + "\n\n---\n\n" + x["solution"]["en"]
        cells.append(md(hints + "\n\n" + fold("参考答案 / Reference solution", sol)))
    for n, c in enumerate(cells):
        c["id"] = f"cell-{n:02d}"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = root / "notebooks" / f"{m['num']}_{k}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", out.relative_to(root))
