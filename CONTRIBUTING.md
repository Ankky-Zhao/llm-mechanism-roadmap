# Contributing / 贡献指南

Corrections and additions are welcome. 欢迎纠错与补充。

All content lives in `data/` and `exercises/`. `index.html` and `notebooks/` are generated, so please do not edit them by hand. 所有内容都在 `data/` 和 `exercises/` 中，`index.html` 与 `notebooks/` 是生成产物，请不要手动修改。

## Before opening a PR / 提交前

```bash
python3 build.py                  # fails loudly on inconsistent data
python3 scripts/run_tests.py      # every solution passes, every starter fails
python3 scripts/make_notebooks.py
```

Commit the regenerated `index.html` and `notebooks/`. CI checks that they are up to date.

## Fix or add text / 修改正文

Edit `data/content.json`. Every user-facing string is `{"zh": "...", "en": "..."}`. Please update both languages, or say in the PR which one is missing.

- Inline math is `$...$`, display math is `$$...$$`. Backslashes are doubled in JSON, for example `"\\frac{a}{b}"`.
- `` `code` `` and `**bold**` are the only other markup.
- Every stage has `why`, `what` and `meaning` (why the stage is needed, what it does, what it means for the LLM as a whole), and every finished module has `why`.
- A derivation is a list of steps. Each step has one formula (`tex`) and a `note` that says what the step does. If a key result comes from a paper or textbook, add `source` with `text` and `url`.

## Add a glossary term / 增加术语

Append to `glossary` in `data/content.json`. Keep the definition to one or two sentences and point `module` to the module that explains the term (or `null`). An optional `link` gives one resource for going deeper and must be a URL that already appears in `data/resources.json`, so that it has been verified. List the term under `terms` of the module that introduces it.

```json
{"term": {"zh": "贪心解码", "en": "greedy decoding"}, "def": {"zh": "…", "en": "…"}, "module": "pipeline"}
```

## Add a resource / 增加资源

Edit `data/resources.json`:

```json
{
  "topic": "autograd",        // an existing module key in data/content.json
  "level": "core",            // intro | core | deep
  "type": "video",            // official | course | video | blog | paper | code | book
  "lang": "en",               // en | zh
  "title": "…",
  "url": "https://…",
  "hours": 1,
  "why": {"zh": "为什么推荐，一句话", "en": "One sentence on why it is worth the time"},
  "checked": "2026-11-25"     // the day you opened the link and confirmed it
}
```

Open the link yourself on the day you submit. Prefer primary sources. 提交当天请亲自打开链接确认，优先一手来源。

## Add an exercise / 增加练习

Create `exercises/<id>/` with `meta.json`, `starter.py`, `tests.py` and `solution.py`, then list `<id>` under the module's `exercises` in `data/content.json`.

- `starter.py` contains the skeleton with `TODO` comments and raises `NotImplementedError` where code is missing.
- `tests.py` is plain `assert` statements that run in the same namespace, right after the implementation. Where possible compare against the PyTorch implementation.
- `solution.py` must pass `tests.py`, and `starter.py` must fail it. `scripts/run_tests.py` enforces both.
- Keep the run time to a few minutes on a CPU. Set `"network": true` in `meta.json` if a model is downloaded, and `"requires": ["other_id"]` if the exercise builds on another one.
- Give at least one hint. Hints go from gentle to specific, and the solution comes last.

## Add an interview question / 增加面试题

Edit `data/questions.json`. Only add questions that you actually found, with a link. 只收录实际找到并能给出链接的题目。

```json
{
  "id": "softmax-grad-01",
  "module": "softmax_grad",
  "region": "cn",                  // cn | intl
  "companies": ["…"],              // leave empty if the source does not name one
  "role": "算法 / Research Engineer / …",
  "source_type": "report",         // report (first-hand) | collection (curated list) | official
  "sources": [{"title": "…", "url": "https://…"}],
  "orig_lang": "zh",               // language of the original wording
  "q": {"zh": "原题表述", "en": "Translation"},
  "a": {"zh": "要点式参考答案", "en": "Key points"}
}
```

Keep the original wording in its own language and translate the other one. Give key terms in both languages in the answer.
