# 文档与图表理解

> 文档并非普通照片。PDF、科学论文、发票或手写表单包含版面布局、表格、图表、脚注、页眉以及语义结构，这些是纯图像理解模型无法捕捉的。在视觉语言模型（Vision-Language Model, VLM）普及之前，技术栈是一套流水线：Tesseract 光学字符识别（Optical Character Recognition, OCR） + LayoutLMv3 + 表格提取启发式规则。VLM 浪潮引入了免 OCR（OCR-free）模型——如 Donut（2022）、Nougat（2023）、DocLLM（2023）——它们能够直接输出结构化标记（Structured Markup）。到了 2026 年，前沿做法仅仅是“将页面图像以 2576px 原生分辨率输入 Claude Opus 4.7”，结构化标记输出便随之免费获得。本课程将梳理文档人工智能（Document AI）演进的三个时代。

**Type:** 构建实践
**Languages:** Python（标准库（Standard Library）、版面感知（Layout-aware）文档解析器骨架）
**Prerequisites:** 第 12 阶段 · 05（LLaVA），第 5 阶段（自然语言处理（Natural Language Processing, NLP））
**Time:** 约 180 分钟

## 学习目标

- 阐述文档人工智能（Document AI）的三个时代：光学字符识别（OCR）流水线、免 OCR（OCR-free）模型、原生视觉语言模型（VLM-native）架构。
- 描述 LayoutLMv3 的三种输入流：文本、版面（边界框（Bounding Box, bbox））、图像块（Image Patches），以及统一掩码（Unified Masking）机制。
- 对比 Donut（免 OCR，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（版面感知生成式模型）与 PaliGemma 2（原生 VLM）。
- 针对新任务（如发票、科学论文、手写表单、中文收据）选择合适的文档模型。

## 核心问题

“理解这份 PDF”看似简单，实则极具挑战。信息分布在以下维度：

- 文本内容（承载 90% 的信息信号）。
- 版面布局（页眉、脚注、侧边栏、双栏排版）。
- 表格（行、列、合并单元格）。
- 插图与图表。
- 手写批注。
- 字体与排版样式（标题与正文的区别）。

原始光学字符识别（OCR）仅能提取文本，而会丢失其余所有信息。一个处理发票的系统必须知道“总计：$1,245”位于右下角，而非出自脚注。

## 核心概念

### 时代 1 — OCR 流水线 (OCR Pipeline)（2021 年之前）

经典技术栈：

1. PDF → 逐页转换为图像。
2. Tesseract（或商业 OCR 引擎）提取文本，并附带逐词边界框 (Bounding Boxes)。
3. 版面分析器 (Layout Analyzer) 识别区块（页眉、表格、段落）。
4. 表格结构识别器 (Table Structure Recognizer) 解析表格。
5. 领域规则 + 正则表达式 (Regex) 提取字段。

适用于清晰的印刷文本。但在手写体、倾斜扫描件、复杂表格或非英文字符集上会失效。每种失败模式都需要定制化的异常处理路径。

### TrOCR（2021 年）

TrOCR（Li 等人，arXiv:2109.10282）使用在合成与真实文本图像上训练的 Transformer 编码器-解码器 (Transformer Encoder-Decoder)，取代了 Tesseract 经典的 CNN-CTC 架构。在手写体和多语言文本上取得了显著优势。它仍然是一个流水线（检测器 → TrOCR → 版面分析），但 OCR 步骤的性能得到了大幅提升。

### 时代 2 — 无 OCR 架构 (OCR-free Models)（2022-2023 年）

首批无 OCR 模型的理念是：完全跳过检测步骤，直接将图像像素映射为结构化输出。

Donut（Kim 等人，arXiv:2111.15664）：
- 采用编码器-解码器 Transformer 架构，编码器为 Swin-B。
- 输出格式灵活：表单理解输出 JSON，摘要生成输出 Markdown，或适配任何任务特定的数据模式 (Schema)。
- 无需 OCR、无需版面分析、无需目标检测。

Nougat（Blecher 等人，arXiv:2308.13418）：
- 专门针对科学论文进行训练。
- 输出为 LaTeX / Markdown。
- 能够处理数学公式、多栏排版和插图。
- 几乎所有 arXiv 解析器都会调用的模型。

这些是专用模型，而非通用模型。用 Donut 处理科学论文会失败；用 Nougat 处理发票也会失败。

### LayoutLMv3（2022 年）

另一条技术路线。LayoutLMv3（Huang 等人，arXiv:2204.08387）保留了 OCR，但引入了版面理解能力：

- 三路输入流：OCR 文本词元 (Tokens)、逐词元的二维边界框、图像块 (Image Patches)。
- 跨三种模态的掩码训练目标 (Masked Training Objective)（掩码文本、掩码图像块、掩码版面）。
- 下游任务：分类、实体抽取、表格问答 (Question Answering, QA)。

LayoutLMv3 代表了基于 OCR 的文档理解技术的巅峰。在表单和发票处理上表现强劲。需要上游提供 OCR 结果。在标准化文档基准测试中，它是 VLM 出现前准确率最高的模型。

### DocLLM（2023 年）

DocLLM（Wang 等人，arXiv:2401.00908）是 LayoutLM 的生成式衍生模型。它基于版面词元生成自由格式的答案。更适用于文档问答；但仍依赖 OCR 输入。

### 时代 3 — 原生 VLM 架构 (VLM-native)（2024 年至今）

2024 年，视觉语言模型 (Vision-Language Models, VLMs) 的性能已足够强大，可以完全取代传统流水线。只需将高分辨率的整页图像输入 VLM，提出问题，即可获取答案。

- LLaVA-NeXT 的 336-tile AnyRes 架构适用于小型文档。
- Qwen2.5-VL 的动态分辨率机制原生支持 2048+ 像素。
- Claude Opus 4.7 支持 2576 像素的文档。
- PaliGemma 2（2025 年 4 月）专门针对文档与手写体进行训练。

原生 VLM 架构与 OCR 流水线之间的差距迅速缩小。到 2026 年，原生 VLM 在以下场景占据优势：

- 场景文本（手写体与印刷体混合、多语种混合）。
- 包含合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带有文本标注的插图。

OCR 流水线在以下场景仍具优势：

- 海量纯扫描件处理，且对单页延迟敏感的场景。
- 流水线可靠性（确定性失败 vs VLM 幻觉 (Hallucinations)）。
- 需要可审计 OCR 输出的合规/监管环境。

### Claude 4.7 / GPT-5 前沿模型

在 2576 像素原生输入下，前沿 VLM 的文档理解准确率已接近人类水平。2026 年初的基准测试数据如下：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，流水线版 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

闭源模型之间的差距主要在于分辨率和基础大语言模型 (Base LLM) 的规模。7B 参数的开源模型落后几个百分点，但正在快速追赶。

### 数学公式与 LaTeX 输出

科学论文需要精确的 LaTeX 公式输出。Nougat 正是为此训练的。使用 LaTeX 作为训练目标的 VLM（如 Qwen2.5-VL-Math、Nougat 衍生模型）能够生成可用的 LaTeX 代码。若未经显式的 LaTeX 训练，VLM 生成的转录结果虽可读但不够精确。

2026 年科学论文处理流水线建议：先用 Nougat 处理 PDF，再对疑难页面使用 VLM 进行补充。

### 手写体识别

这仍然是最具挑战性的子任务。在印刷体与手写体混合的场景（如医生处方、已填写的表单）中，出于成本考量，OCR 流水线仍优于 VLM。纯手写体识别的 VLM 正在不断进步（如 Claude 4.7、PaliGemma 2）。

### 2026 年实践指南

针对全新的文档 AI 项目：

- 海量纯印刷发票：LayoutLMv3 + 规则引擎，成本效益高。
- 混合文档（科学论文 + 手写体 + 表单）：原生 VLM 架构（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 数据摄入：Nougat 处理数学公式，VLM 处理插图。
- 合规/监管场景：OCR 流水线 + VLM 校验器进行交叉验证 (Cross-check)。

## 使用它

`code/main.py`:

- 一个简易的布局感知分词器（layout-aware tokenizer）：接收（文本，边界框（bbox））对，生成类似 LayoutLMv3 的输入。
- 一个 Donut 风格的任务模式生成器（task schema generator）：用于表单的 JSON 模板。
- 对比了 OCR 流水线（OCR pipeline）、Donut、Nougat 和原生视觉语言模型（VLM-native）方案每页的 Token 预算（token budgets）。

## 交付成果

本课时将生成 `outputs/skill-document-ai-stack-picker.md`。针对文档 AI（document-AI）项目（需综合评估领域、规模、质量与合规性要求），在 OCR 流水线、免 OCR 专用模型（OCR-free specialist）和原生视觉语言模型之间进行选型。

## 练习

1. 你的项目每天处理 1000 万张发票。在不损失准确率的前提下，哪种技术栈能最小化单页成本？

2. 为什么 LayoutLMv3 在表单问答（form QA）上优于纯 CLIP 视觉语言模型（pure-CLIP-VLMs），但在场景文本（scene-text）上表现较差？边界框流（bbox stream）牺牲了什么？

3. Nougat 生成 LaTeX 代码。请提出一个测试用例，说明在 LaTeX 保真度（LaTeX fidelity）上 VLM-native 的输出优于 Nougat；再提出一个 Nougat 胜出的用例。

4. 阅读 PaliGemma 2 论文（Google, 2024）。相较于 PaliGemma 1，是增加了哪项关键训练数据提升了文档准确率？

5. 设计一个符合监管要求的混合架构（regulatory-safe hybrid）：以 OCR 流水线为主，视觉语言模型（VLM）为辅进行交叉验证。当两者结果不一致时，如何解决分歧？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| OCR 流水线 | “Tesseract 风格” | 分阶段技术栈：检测 -> OCR -> 版面分析 -> 规则；确定性高但脆弱 |
| 免 OCR | “Donut 风格” | 跳过显式 OCR 步骤的图像到输出 Transformer；单模型架构 |
| 布局感知 | “LayoutLM” | 输入包含每个 Token 的边界框坐标；跨模态统一掩码 |
| 原生视觉语言模型 | “前沿 VLM” | 将高分辨率页面图像直接输入 Claude/GPT/Qwen 等视觉语言模型；无需流水线 |
| DocVQA | “文档基准” | 文档视觉问答标准；引用率最高的评分指标 |
| 标记语言输出 | “LaTeX / Markdown” | 结构化输出格式而非自由文本；支持下游自动化处理 |

## 延伸阅读

- [Li 等人 — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher 等人 — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang 等人 — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim 等人 — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang 等人 — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)