# 大模型机制学习路线 · LLM Mechanism Roadmap

[中文](#中文) | [English](#english)

🌐 **在线页面 / Live page** · https://ankky-zhao.github.io/llm-mechanism-roadmap/

一条中英双语的学习路线，用于理解大语言模型的内部机制。路线从单个标量的反向传播开始，到 activation patching 结束，共 10 个阶段、37 个模块。

A bilingual (Chinese and English) roadmap for understanding the internal mechanisms of large language models. It starts with backpropagation on a single scalar and ends with activation patching, in 10 stages and 37 modules.

| | |
|---|---|
| 模块 / Modules | 37（10 个阶段 / 10 stages） |
| 练习 / Exercises | 114（代码练习与推导练习 / coding and derivation） |
| 推荐资源 / Resources | 222（每条附推荐理由 / each with a reason） |
| 术语 / Glossary terms | 198 |
| “为什么”问题 / Why-questions | 190 余个 / more than 190 |
| 面试题 / Interview questions | 43（检索于 2026-11-25 / retrieved on 2026-11-25） |
| Notebook | 37（每个模块一个，可在 Colab 打开 / one per module, opens in Colab） |
| 预计学习时间 / Estimated time | 约 81 小时 / about 81 hours |

姊妹路线 / Sibling roadmaps · [Agent Atlas](https://github.com/Ankky-Zhao/llm-agent-roadmap)（LLM agents）· [Web Scraping Roadmap](https://github.com/Ankky-Zhao/web-scraping-roadmap)

---

## 中文

### 目的

这条路线围绕三个目标设计。

- **理解机制与推导。** 模型的每个部件为什么这样设计，损失函数和梯度怎样得到，这些内容以逐步推导的形式给出，每一步配一句说明，关键结论标注论文或教材出处。
- **回答“为什么”类问题。** 大厂面试常见的问题多数属于这一类，例如注意力为什么要除以 $\sqrt{d_k}$、为什么用 LayerNorm 而较少用 BatchNorm、KV cache 为什么能加速。路线为每个模块整理了这类问题，并另设面试题库，覆盖国内与海外公司。
- **学会观察模型内部。** 阶段 9 介绍可解释性的常用方法，包括注意力可视化、logit lens、activation patching、线性探针和引导向量，使“模型为什么这样回答”成为可以用实验检验的问题。

### 适合的读者

- 用过大模型的 API 或做过应用，希望了解内部原理的学习者
- 准备算法岗或大模型相关岗位面试的求职者
- 研究中需要解释模型行为的研究生，特别是可解释性与 agent 方向

前置知识为 Python 基础、线性代数和微积分的入门内容。阶段 1 会复习后续用到的数学。

### 路线的组织方式

路线采用总分结构。阶段 0 用一张流程图把一次生成从头到尾走一遍，后面九个阶段各展开其中一段。每个阶段的开头回答三个问题，即为什么需要这个阶段、它在做什么、它对整个大模型的意义。

| 阶段 | 内容 | 模块 |
|---|---|---|
| 00 总览 | 用一句话把一次生成从头到尾走一遍，再交代训练和推理两条时间线 | 0.1 一次生成的全过程 · 0.2 模型的一生与路线设计 |
| 01 基础与数学准备 | 手写反向传播，从最大似然推出交叉熵损失，推出 softmax 的梯度 | 1.1 张量与自动求导 · 1.2 交叉熵、最大似然与困惑度 · 1.3 softmax 及其梯度 |
| 02 Transformer 架构 | 把一层拆成部件逐个实现，最后分析计算代价 | 2.1 缩放点积注意力 · 2.2 多头注意力与参数量 · 2.3 因果掩码与三类架构 · 2.4 位置编码 · 2.5 归一化、残差与 FFN · 2.6 注意力的复杂度 |
| 03 分词与上下文 | 手写 BPE 分词，分析上下文窗口的限制，观察分词引起的典型问题 | 3.1 BPE 与 SentencePiece · 3.2 上下文窗口 · 3.3 分词带来的问题 |
| 04 训练原理 | 依次推导预训练、指令微调、RLHF、DPO 的损失函数，实现 LoRA，讨论缩放定律 | 4.1 预训练目标与数据 · 4.2 指令微调 · 4.3 RLHF · 4.4 DPO · 4.5 LoRA · 4.6 缩放定律与涌现 |
| 05 推理与解码 | 实现各种采样策略、KV cache 和保证输出格式的约束解码 | 5.1 采样策略 · 5.2 KV cache 与 prefill/decode · 5.3 结构化输出与约束解码 |
| 06 模型与工具生态 | 比较闭源模型与开放权重模型，梳理常见许可证允许和禁止的用途，介绍 MLOps 的环节和 MLflow 等工具 | 6.1 闭源与开放权重模型 · 6.2 许可证与使用限制 · 6.3 MLOps 与 LLMOps 工具 |
| 07 应用模式 | 不依赖框架实现 prompt 技巧、RAG（含分块与向量索引）和 agent 循环，再与框架实现对比 | 7.1 提示工程与上下文学习 · 7.2 RAG · 7.3 智能体系统 |
| 08 评估与负责任使用 | 实现常用指标和 LLM 评委，分析幻觉、偏见与隐私，介绍欧盟、中国和美国的监管要求 | 8.1 评估方法 · 8.2 幻觉、偏见与隐私 · 8.3 欧盟人工智能法 · 8.4 中国与美国的监管 |
| 09 可解释性与机制分析 | 实现注意力可视化、logit lens、activation patching、线性探针和引导向量 | 9.1 注意力模式可视化 · 9.2 Logit lens 与直接 logit 归因 · 9.3 激活修补 · 9.4 线性探针与引导向量 |

### 每个模块包含的内容

| 部分 | 说明 |
|---|---|
| 为什么要学 | 这个模块解决哪个环节的问题，不学会在后面遇到什么困难 |
| 打个比方 | 用日常例子说明机制的核心想法，再进入正式定义 |
| 本模块的新术语 | 首次出现的术语，均收录在术语表中 |
| 讲解与示意图 | 中英双语的正文和一张示意图 |
| 逐步推导 | 公式用 KaTeX 渲染，每一步配说明，关键结论标注出处 |
| 推荐资源 | 论文、课程、博客和代码，每条说明推荐理由，链接经过逐条访问核实 |
| 代码练习 | 给出函数框架和 TODO，用 `assert` 或与 PyTorch 官方实现对比来检查，先给提示再给参考答案 |
| 推导练习 | 需要动手推的小题，附答案 |
| 常见疑问 | 学习时容易卡住的问题和面试中的追问 |

### 页面上的其他栏目

- **“为什么”专栏。** 汇总各模块的“为什么”问题，共 190 余个，可按阶段浏览。
- **面试题库。** 目前收录 43 题，只收录实际检索到的题目，不自行编写。每题标注公司、岗位、地区和来源链接，并区分原始面经与整理合集。题目和参考答案均为中英双语，可按公司、地区和模块筛选。
- **术语表。** 198 个术语，每个术语指向解释它的模块。
- **学习进度。** 勾选记录保存在浏览器本地（localStorage），不上传数据。

### 如何使用

**在线阅读。** 打开[在线页面](https://ankky-zhao.github.io/llm-mechanism-roadmap/)，右上角可切换中文和英文。建议先读阶段 0，再按顺序学习，已有基础的读者可以直接跳到感兴趣的阶段。

**做练习。** 每个模块页面上有 Colab 链接，打开后可直接运行。本地运行的方式如下。

```bash
pip install -r requirements.txt
# 补全 exercises/<id>/starter.py 中的 TODO，再把 tests.py 的内容接在后面运行
python3 scripts/run_tests.py                  # 运行全部练习的参考答案与框架
python3 scripts/run_tests.py --skip-network   # 跳过需要下载模型的练习
```

练习使用小型开放权重模型（如 GPT-2 small），在 CPU 或免费 Colab 上即可运行，不需要付费 API。

**修改内容并重新生成页面。**

```bash
python3 build.py                    # 由 data/ 与 exercises/ 生成 index.html，无第三方依赖
python3 scripts/make_notebooks.py   # 重新生成 notebooks/
python3 scripts/check_links.py      # 检查资源链接
```

### 仓库结构

```
data/
  content.json     阶段、模块正文、推导步骤、常见疑问、术语表
  resources.json   推荐资源及推荐理由
  questions.json   面试题及来源
  figures/*.svg    示意图
exercises/<id>/
  meta.json        标题、题面、提示
  starter.py       带 TODO 的函数框架
  tests.py         自动检查
  solution.py      参考答案
notebooks/         每个模块一个 .ipynb，由脚本生成
scripts/           run_tests.py、make_notebooks.py、check_links.py
template.html      页面模板
build.py           构建脚本
index.html         生成的页面，用于 GitHub Pages
```

### 部署

仓库通过 GitHub Actions 发布到 GitHub Pages。在 Settings → Pages 中把 Source 设为 **GitHub Actions** 后，`.github/workflows/pages.yml` 会在每次 push 到 `main` 时构建并发布。`tests.yml` 在 push 和 PR 时运行练习，`link-check.yml` 每周检查一次链接。

### 说明与局限

- 参考答案在编写时均实际运行并通过测试，未完成的框架确认不能通过测试。不同版本的 `torch` 和 `transformers` 可能使个别数值略有差异。
- 阶段 9 的实验在小模型上完成，样本量有限，结论用于说明方法，不宜直接推广到大模型。
- 模型生态、许可证和监管内容具有时效性，页面上标注了核对日期。监管模块仅供学习参考，不构成法律意见。
- 面试题来自公开的面经和合集，无法逐一核实其真实性，题库中标注了来源类型供读者判断。

### 贡献

欢迎纠错与补充，见 [CONTRIBUTING.md](CONTRIBUTING.md)。

### 许可

- 内容（`data/`、`exercises/*/meta.json`、页面文案）采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 代码（`template.html`、`build.py`、`scripts/`、练习中的 `.py` 文件）采用 MIT 许可

资源与论文的版权归各自作者所有。

---

## English

### Purpose

The roadmap is designed around three goals.

- **Understanding mechanisms and derivations.** Why each component is designed the way it is, and how the losses and gradients are obtained, is presented as step-by-step derivations. Each step carries a short note, and key results cite the paper or textbook they come from.
- **Answering why-questions.** Many interview questions at large technology companies are of this kind, such as why attention is divided by $\sqrt{d_k}$, why LayerNorm is preferred over BatchNorm, or why a KV cache speeds up decoding. Such questions are collected for each module, and a separate interview bank covers companies in China and abroad.
- **Learning to look inside a model.** Stage 9 introduces common interpretability methods, including attention visualization, the logit lens, activation patching, linear probes and steering vectors, so that the question of why a model answered as it did becomes one that experiments can test.

### Who it is for

- Learners who have used LLM APIs or built applications and would like to know what happens inside
- Candidates preparing for machine learning or LLM-related interviews
- Graduate students who need to explain model behaviour in their research, especially in interpretability and agents

The prerequisites are basic Python, introductory linear algebra and calculus. Stage 1 reviews the mathematics used later.

### How the roadmap is organised

The whole is presented before the parts. Stage 0 walks one generation from start to finish with a single flow chart, and each of the following nine stages expands one segment of it. Every stage opens with three questions, which are why the stage is needed, what it does, and what it means for the LLM as a whole.

| Stage | What it does | Modules |
|---|---|---|
| 00 The big picture | Walks one sentence through a full generation, then lays out the two timelines of training and inference | 0.1 One generation, start to finish · 0.2 The life of a model and the design of this roadmap |
| 01 Foundations and math | Writes backpropagation by hand, derives the cross-entropy loss from maximum likelihood, and derives the gradient of softmax | 1.1 Tensors and automatic differentiation · 1.2 Cross-entropy, maximum likelihood and perplexity · 1.3 Softmax and its gradient |
| 02 Transformer architecture | Takes a layer apart, implements each component, then analyses the cost | 2.1 Scaled dot-product attention · 2.2 Multi-head attention and parameter counts · 2.3 Causal masking and the three architectures · 2.4 Positional encodings · 2.5 Normalization, residuals and the FFN · 2.6 Attention complexity |
| 03 Tokenization and context | Writes BPE tokenization by hand, analyses the limits of the context window and observes typical problems caused by tokenization | 3.1 BPE and SentencePiece · 3.2 The context window · 3.3 Problems caused by tokenization |
| 04 How models are trained | Derives the losses of pretraining, instruction tuning, RLHF and DPO in turn, implements LoRA and discusses scaling laws | 4.1 Pretraining objective and data · 4.2 Instruction tuning · 4.3 RLHF · 4.4 DPO · 4.5 LoRA · 4.6 Scaling laws and emergence |
| 05 Inference and decoding | Implements the sampling strategies, the KV cache and constrained decoding that guarantees the output format | 5.1 Sampling strategies · 5.2 KV cache, prefill and decode · 5.3 Structured output and constrained decoding |
| 06 Models and tooling | Compares closed and open-weight models, sorts out what common licenses allow and forbid, and introduces the stages of MLOps and tools such as MLflow | 6.1 Closed and open-weight models · 6.2 Licenses and usage restrictions · 6.3 MLOps and LLMOps tooling |
| 07 Application patterns | Implements prompting techniques, RAG (with chunking and vector indexes) and an agent loop without a framework, then compares with framework versions | 7.1 Prompt engineering and in-context learning · 7.2 RAG · 7.3 Agent systems |
| 08 Evaluation and responsible use | Implements common metrics and an LLM judge, analyses hallucination, bias and privacy, and introduces regulation in the European Union, China and the United States | 8.1 Evaluation methods · 8.2 Hallucination, bias and privacy · 8.3 The EU AI Act · 8.4 Regulation in China and the United States |
| 09 Interpretability | Implements attention visualization, the logit lens, activation patching, linear probes and steering vectors | 9.1 Visualizing attention · 9.2 The logit lens and direct logit attribution · 9.3 Activation patching · 9.4 Linear probes and steering vectors |

### What each module contains

| Part | Description |
|---|---|
| Why learn this | Which problem the module addresses and what becomes difficult later without it |
| An everyday picture | The core idea of the mechanism in an everyday example, before the formal definition |
| New terms | Terms that appear for the first time, each listed in the glossary |
| Explanation and figure | Bilingual text and one diagram |
| Step-by-step derivations | Formulas rendered with KaTeX, one note per step, sources cited for key results |
| Recommended resources | Papers, courses, blog posts and code, each with a reason, and each link visited and checked |
| Coding exercises | A function skeleton with TODOs, checked by `assert` or against the official PyTorch implementation, with hints before the reference solution |
| Derivation exercises | Short problems to work out by hand, with answers |
| FAQ | Points where learners tend to get stuck and follow-up questions from interviews |

### Other sections on the page

- **Why column.** The why-questions of the modules in one place, more than 190 in total, browsable by stage.
- **Interview bank.** 43 questions at present. Only questions that were actually found are included, and none were written for the bank. Each question lists the company, role, region and source links, and is labelled as either a first-hand interview report or a curated collection. Questions and reference answers are bilingual and can be filtered by company, region and module.
- **Glossary.** 198 terms, each pointing to the module that explains it.
- **Progress.** Checkmarks are stored locally in the browser (localStorage). No data is uploaded.

### How to use it

**Read online.** Open the [live page](https://ankky-zhao.github.io/llm-mechanism-roadmap/) and switch between Chinese and English in the top right corner. Reading stage 0 and then following the order is recommended. Readers with some background may jump to the stages of interest.

**Do the exercises.** Each module page has a Colab link that runs as it is. To run locally, use the following commands.

```bash
pip install -r requirements.txt
# fill in the TODOs in exercises/<id>/starter.py, then append tests.py and run
python3 scripts/run_tests.py                  # run the solutions and starters of the exercises
python3 scripts/run_tests.py --skip-network   # skip exercises that download a model
```

The exercises use small open-weight models such as GPT-2 small and run on a CPU or on free Colab. No paid API is needed.

**Edit the content and rebuild the page.**

```bash
python3 build.py                    # builds index.html from data/ and exercises/, no third-party dependency
python3 scripts/make_notebooks.py   # regenerates notebooks/
python3 scripts/check_links.py      # checks the resource links
```

### Repository layout

```
data/
  content.json     stages, module text, derivation steps, FAQ, glossary
  resources.json   recommended resources with reasons
  questions.json   interview questions with sources
  figures/*.svg    diagrams
exercises/<id>/
  meta.json        title, statement, hints
  starter.py       function skeleton with TODOs
  tests.py         automatic checks
  solution.py      reference solution
notebooks/         one .ipynb per module, generated by a script
scripts/           run_tests.py, make_notebooks.py, check_links.py
template.html      page template
build.py           build script
index.html         generated page, served by GitHub Pages
```

### Deployment

The repository is published to GitHub Pages through GitHub Actions. Once Source is set to **GitHub Actions** under Settings → Pages, `.github/workflows/pages.yml` builds and publishes on each push to `main`. `tests.yml` runs the exercises on pushes and pull requests, and `link-check.yml` checks the links once a week.

### Notes and limitations

- The reference solutions were run and passed their tests when they were written, and the unfinished starters were confirmed to fail. Different versions of `torch` and `transformers` may change individual numbers slightly.
- The experiments of stage 9 were carried out on small models with limited sample sizes. The conclusions illustrate the methods and should not be generalised directly to large models.
- The content on the model ecosystem, licenses and regulation is time-sensitive, and the page states the date on which it was checked. The regulation modules are for learning purposes and do not constitute legal advice.
- The interview questions come from public interview reports and collections, and their authenticity cannot be verified one by one. The source type is labelled so that readers can judge.

### Contributing

Corrections and additions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

### License

- Content (`data/`, `exercises/*/meta.json`, page text) is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Code (`template.html`, `build.py`, `scripts/`, the `.py` files of the exercises) is released under the MIT License

The copyright of the listed resources and papers remains with their authors.

---

Curated by Anqi (Ankky) Zhao · Technical University of Munich
