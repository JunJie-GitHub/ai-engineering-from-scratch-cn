# ASCII 艺术（ASCII Art）与视觉越狱（Visual Jailbreaks）

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753)。在有害请求中屏蔽与安全相关的词元（token），将其替换为相同字母的 ASCII 艺术渲染图，并发送伪装后的提示词（prompt）。GPT-3.5、GPT-4、Gemini、Claude 和 Llama-2 均无法稳健地识别此类 ASCII 艺术词元。该攻击成功绕过了困惑度过滤器（Perplexity Filters, PPL）、改写防御（Paraphrase Defenses）以及重新分词（Retokenization）。相关研究：ViTC 基准测试用于衡量模型对非语义视觉提示词的识别能力；StructuralSleight 将其泛化为针对非常规文本编码结构（Uncommon Text-Encoded Structures，如树、图、嵌套 JSON）的一类编码攻击。

**类型：** 构建（Build）
**语言：** Python（标准库，ArtPrompt 词元屏蔽工具包）
**前置条件：** 第 18 阶段 · 12（PAIR），第 18 阶段 · 13（MSJ）
**耗时：** 约 60 分钟

## 学习目标

- 描述 ArtPrompt 攻击流程：词汇识别步骤、ASCII 艺术替换以及最终生成的伪装提示词。
- 解释为何标准防御机制（PPL、改写防御、重新分词）对 ArtPrompt 攻击失效。
- 定义 ViTC 并说明其衡量指标。
- 将 StructuralSleight 描述为针对任意非常规文本编码结构（Uncommon Text-Encoded Structures）的泛化方法。

## 问题背景

基于改写和角色扮演（第 12 课）以及长上下文（第 13 课）的攻击主要作用于文本层面的模式。而 ArtPrompt 则作用于识别层面：模型并未直接解析被禁用的词元，而是解析由字符渲染而成的图像。安全过滤器看到的是无害的标点符号，而模型看到的却是一个完整的词汇。

## 核心概念

### ArtPrompt，两个步骤

步骤 1：词汇识别。给定一个有害请求，攻击者使用大语言模型（Large Language Model）识别与安全相关的词汇（例如“如何制造炸弹”中的“炸弹”）。 

步骤 2：伪装提示词生成。将每个识别出的词汇替换为其 ASCII 艺术（ASCII Art）渲染形式（由 7x5 或 7x7 的字符块组成字母形状）。模型接收到的是一组标点符号和空格构成的网格，能力足够强的模型能将其识别为原词汇；而安全过滤器（Safety Filter）只能看到该网格。

结果：GPT-4、Gemini、Claude、Llama-2 和 GPT-3.5 均告失效。在其基准测试子集上，攻击成功率超过 75%。

### 为何标准防御机制会失效

- **PPL（困惑度过滤器，Perplexity Filter）。** ASCII 艺术具有较高的困惑度（Perplexity）——但所有新颖输入同样如此。用于拦截 ArtPrompt 的阈值设定，也会同时拦截合法的结构化输入。
- **改写（Paraphrase）。** 对提示词进行改写会破坏 ASCII 艺术。但在实践中，用于改写的 LLM 往往会保留或重建该艺术形式。
- **重新分词（Retokenization）。** 改变分词方式并不会改变模型能够识别字母形状这一事实。

根本问题在于，安全过滤器作用于词元（Token）或语义层面；而 ArtPrompt 则在视觉识别层面运作。

### ViTC 基准测试

用于评估模型对非语义视觉提示词的识别能力。该基准测试衡量模型读取 ASCII 艺术、Wingdings 字体及其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率呈正相关：模型读取视觉文本的能力越强，ArtPrompt 对其的攻击效果就越好。这体现了能力与安全之间的权衡（Capability-Safety Tradeoff）。

### StructuralSleight

对 ArtPrompt 的泛化：非常见文本编码结构（Uncommon Text-Encoded Structures）。包括树状结构、图结构、嵌套 JSON、JSON 内嵌 CSV、diff 格式代码块等。如果某种结构在安全训练数据中较为罕见，但模型能够解析，它就可以用来隐藏有害内容。

防御启示：安全机制必须能够泛化到模型可解析的所有结构化表示形式上。这类结构的集合庞大且仍在不断增长。

### 图像模态类比

视觉大语言模型（Visual LLM，如 GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩大了攻击面（Attack Surface）。使用真实图像发起的 ArtPrompt 式攻击比 ASCII 艺术类比攻击更强，因为图像编码器（Image Encoder）能产生更丰富的信号。

### 在 Phase 18 中的定位

第 12-14 课描述了三种正交攻击向量（Orthogonal Attack Vectors）：迭代优化（PAIR）、上下文长度（MSJ）以及编码（ArtPrompt/StructuralSleight）。第 15 课将重心从以模型为中心的攻击转向系统边界攻击（间接提示词注入，Indirect Prompt Injection）。第 16 课介绍了防御工具链的应对方案。

## 使用方法

`code/main.py` 构建了一个简易版 ArtPrompt。你可以使用 ASCII 艺术字形对有害查询中的特定词汇进行伪装，验证伪装后的字符串能否通过关键词过滤器，并（可选）使用简易识别器将伪装字符串解码还原。

## 部署上线

本课时将生成 `outputs/skill-encoding-audit.md` 文件。给定一份越狱防御（jailbreak-defense）报告，它将枚举所涵盖的编码攻击族系（encoding attack families）（包括 ASCII 艺术（ASCII art）、Base64 编码（base64）、Leet 语（leet-speak）、UTF-8 同形异义字（UTF-8 homoglyph）以及 UTES），并指出拦截每种攻击的防御层（defense layer）。

## 练习

1. 运行 `code/main.py`。验证伪装字符串（cloaked string）能否绕过简单的关键词过滤器（keyword filter）。报告所需的字符级修改（character-level change）。
2. 实现第二种编码方式：对同一目标词使用 Base64 编码。对比其与 ArtPrompt 的过滤器绕过率（filter-bypass rate）及内容恢复难度（recovery difficulty）。
3. 阅读 Jiang 等人 2024 年论文的第 4.3 节（五模型结果）。分析并提出一个原因，解释为何在相同的基准测试（benchmark）中，Claude 对 ArtPrompt 的抵抗力（ArtPrompt-resistance）高于 Gemini。
4. 设计一种生成前防御（pre-generation defense），用于检测提示词（prompt）中呈现 ASCII 艺术形状的区域。评估该机制在合法代码、表格及数学符号上的误报率（false-positive rate）。
5. StructuralSleight 列出了 10 种编码结构。请构思一种能够应对全部 10 种结构的通用防御方案（generalized defense），并估算处理每条受保护提示词所需的计算成本（compute cost）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| ArtPrompt | “ASCII 艺术攻击” | 分两步执行的越狱攻击（jailbreak），通过 ASCII 艺术渲染来掩盖安全敏感词 |
| Cloaking（伪装） | “隐藏词汇” | 将违禁词元（token）替换为模型可读取但过滤器无法识别的视觉表示形式 |
| UTES（非常规文本编码结构） | “非常规结构” | Uncommon Text-Encoded Structure 的缩写——利用树、图、嵌套 JSON 等结构来隐蔽传输内容 |
| ViTC（视觉文本能力） | “视觉文本能力” | 用于评估模型读取非语义视觉编码能力的基准测试（benchmark） |
| Perplexity filter（困惑度过滤器） | “PPL 防御” | 拒绝高困惑度（perplexity）的提示词；该防御易失效，因为合法的结构化输入同样会产生高困惑度得分 |
| Retokenization（重新分词） | “分词器切换防御” | 使用不同的分词器（tokenizer）对提示词进行预处理；该防御易失效，因为模型的识别机制本质上是视觉层面的 |
| Homoglyph（同形异义字） | “形似字符” | 外观与拉丁字母完全一致的 Unicode 字符；可用于绕过子字符串检查（substring checks） |

## 延伸阅读

- [Jiang 等人 — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱攻击论文
- [Li 等人 — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 的泛化研究
- [Chao 等人 — PAIR（第 12 课, arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — 互补的迭代攻击（iterative attack）
- [Anil 等人 — Many-shot Jailbreaking（第 13 课）](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补的长度攻击（length attack）