# 自然语言推理（Natural Language Inference）—— 文本蕴含（Textual Entailment）

> “t entails h”表示人类在阅读前提（premise）`t` 后会推断假设（hypothesis）`h` 为真。自然语言推理（NLI）的任务是预测文本间的蕴含（entailment）、矛盾（contradiction）或中立（neutral）关系。该任务表面看似枯燥，但在实际生产系统中却起着核心支撑作用。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 5 阶段 · 05（情感分析），第 5 阶段 · 13（问答系统）
**Time:** 约 60 分钟

## 核心问题

你构建了一个文本摘要模型。它生成了一段摘要。你如何确保摘要中不包含幻觉（hallucination）？

你开发了一个聊天机器人。它回答了“是”。你如何确认该答案确实得到了检索段落的支持？

你需要对 10,000 篇新闻文章进行主题分类。但你没有任何训练标签。能否直接复用现有模型？

这三个问题最终都可归结为自然语言推理。NLI 的核心问题是：给定前提 `t` 和假设 `h`，`h` 是被 `t` 蕴含、矛盾，还是中立（无关）？

- **幻觉检测：** `t` = 源文档，`h` = 摘要声明。非蕴含即代表存在幻觉。
- **基于事实的问答（Grounded QA）：** `t` = 检索到的段落，`h` = 生成的答案。非蕴含即代表答案系捏造。
- **零样本分类（Zero-shot classification）：** `t` = 文档，`h` = 口语化标签（如“本文关于体育”）。蕴含关系即对应预测标签。

一项任务，三种生产级应用场景。这也是为什么每个检索增强生成（Retrieval-Augmented Generation, RAG）评估框架底层都会内置一个 NLI 模型。

## 核心概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三个标签类别。**

- **蕴含（Entailment）。** `t` → `h`。“猫在垫子上”蕴含“有一只猫”。
- **矛盾（Contradiction）。** `t` → ¬`h`。“猫在垫子上”与“没有猫”相矛盾。
- **中立（Neutral）。** 无法进行双向推断。“猫在垫子上”与“猫饿了”呈中立关系。

**并非逻辑蕴含。** NLI 属于*自然*语言推理——它关注的是普通人类读者会如何推断，而非严格的数理逻辑。在 NLI 中，“约翰遛了他的狗”蕴含“约翰有一只狗”；但在严格的一阶逻辑（first-order logic）中，除非你将“拥有关系”公理化，否则无法直接推导。

**数据集。**

- **SNLI**（2015）。包含 57 万个人工标注的文本对，以图像描述作为前提。领域较为单一。
- **MultiNLI**（2017）。涵盖 10 种体裁的 43.3 万个文本对。截至 2026 年仍是标准训练语料库。
- **ANLI**（2019）。对抗性 NLI 数据集。由人类专门编写旨在攻破现有模型的样本。难度更高。
- **DocNLI、ConTRoL**（2020–2021）。前提为文档长度。用于测试多跳推理（multi-hop inference）与长程推理能力。

**模型架构。** Transformer 编码器（Transformer encoder，如 BERT、RoBERTa、DeBERTa）会读取 `[CLS] premise [SEP] hypothesis [SEP]` 格式的输入。`[CLS]` 标记的向量表示随后输入到一个三分类 softmax 层中。在 MNLI 上训练，在预留基准测试集上评估，即可在分布内（in-distribution）文本对上取得 90% 以上的准确率。

**基于 NLI 的零样本分类。** 给定一篇文档和一组候选标签，将每个标签转化为假设语句（例如“本文关于体育”）。分别计算每个假设的蕴含概率，并选取最大值。这正是 Hugging Face 的 `zero-shot-classification` 管道（pipeline）背后的核心机制。

## 动手实践

### 步骤 1：运行预训练的自然语言推理（Natural Language Inference, NLI）模型

from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]

在生产环境中部署 NLI 时，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是默认的开源选择。DeBERTa-v3 在各大排行榜上名列前茅。

### 步骤 2：零样本分类（Zero-Shot Classification）

zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}

默认使用的提示模板为 "This example is about {label}."。你可以通过 `hypothesis_template` 参数进行自定义。无需训练数据，也无需微调（Fine-tuning），开箱即用。

### 步骤 3：检索增强生成（Retrieval-Augmented Generation, RAG）的忠实度检查

def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold

这是 RAGAS 忠实度评估的核心逻辑。将生成的回答拆分为原子级主张（Atomic Claims），逐一与检索到的上下文进行比对，并计算其中被上下文所蕴含（Entailment）的比例。

### 步骤 4：手动实现的 NLI 分类器（概念演示）

请参阅 `code/main.py` 查看一个仅使用标准库的示例代码：它通过词汇重叠（Lexical Overlap）和否定检测来比较前提（Premise）与假设（Hypothesis）。虽然其性能无法与 Transformer 模型相提并论，但它清晰地展示了该任务的基本形态：输入两段文本，输出三分类标签，损失函数为基于 `{entail, contradict, neutral}` 的交叉熵（Cross-Entropy）。

## 常见陷阱

- **仅依赖假设的捷径（Hypothesis-only Shortcuts）。** 在 SNLI 数据集上，模型仅凭假设文本就能达到约 60% 的预测准确率，因为 "not"、"nobody"、"never" 等词与矛盾（Contradiction）标签高度相关。这是检测标签泄露（Label Leakage）的强基线方法。
- **词汇重叠启发式规则（Lexical Overlap Heuristic）。** 子序列启发式规则（“每个子序列都被蕴含”）能通过 SNLI 测试，但在 HANS/ANLI 数据集上会失效。请使用对抗性基准（Adversarial Benchmarks）进行评估。
- **长文档性能衰减（Document-length Degradation）。** 针对单句训练的 NLI 模型在处理文档级长度的前提时，F1 分数（F1 Score）会下降 20 以上。处理长上下文时，请使用经过 DocNLI 训练的模型。
- **零样本模板敏感性（Zero-shot Template Sensitivity）。** 使用 "This example is about {label}"、"{label}" 或 "The topic is {label}" 等不同模板，可能导致准确率波动超过 10 个百分点。请仔细调优模板。
- **领域不匹配（Domain Mismatch）。** MNLI 模型基于通用英语语料训练。法律、医疗和科学文本需要使用特定领域的 NLI 模型（例如 SciNLI、MedNLI）。

## 开始使用

2026 年技术栈：

| 使用场景 | 模型 |
|---------|-------|
| 通用自然语言推理（Natural Language Inference, NLI） | `microsoft/deberta-v3-large-mnli` |
| 快速推理 / 边缘计算（Edge） | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（Zero-shot Classification）（轻量级） | `facebook/bart-large-mnli` |
| 文档级自然语言推理 | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言（Multilingual） | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| 检索增强生成（Retrieval-Augmented Generation, RAG）中的幻觉检测（Hallucination Detection） | RAGAS / DeepEval 内部的 NLI 层 |

2026 年的核心模式：自然语言推理（NLI）是文本理解的“万能胶”。每当你需要判断“A 是否支持 B？”或“A 是否与 B 矛盾？”时，在调用另一个大语言模型（Large Language Model, LLM）之前，优先考虑使用 NLI。

## 交付上线

保存为 `outputs/skill-nli-picker.md`：

---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.

## 练习

1. **简单。** 在 20 个手工构建的（前提 premise、假设 hypothesis、标签 label）三元组上运行 `facebook/bart-large-mnli`，覆盖全部三个类别。测量准确率。加入对抗性的“子序列启发式”陷阱（例如 "I did not eat the cake" 与 "I ate the cake"），观察模型是否会失效。
2. **中等。** 在 100 条 AG News 标题上，对比零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"` 的效果。报告准确率的波动情况。
3. **困难。** 构建一个 RAG 忠实度（Faithfulness）检查器：原子主张分解 + 逐主张进行 NLI 判断。在 50 个带有标准上下文（gold context）的 RAG 生成答案上进行评估。与人工标注对比，测量假阳性率和假阴性率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | 自然语言推理（Natural Language Inference） | 对前提与假设关系的三分类任务。 |
| RTE | 文本蕴含识别（Recognizing Textual Entailment） | NLI 的旧称；任务相同。 |
| Entailment（蕴含） | “t 蕴含 h” | 普通读者在给定 t 的情况下会认为 h 为真。 |
| Contradiction（矛盾） | “t 排除 h” | 普通读者在给定 t 的情况下会认为 h 为假。 |
| Neutral（中立） | “未定” | 无法从 t 推断出 h 的真假。 |
| Zero-shot classification（零样本分类） | 将 NLI 用作分类器 | 将标签表述为假设，选择蕴含概率最高的标签。 |
| Faithfulness（忠实度） | 答案是否有依据？ | 对（检索到的上下文，生成的答案）进行 NLI 判断。 |

## 延伸阅读

- [Bowman 等人 (2015)。用于学习自然语言推理（Natural Language Inference, NLI）的大规模标注语料库](https://arxiv.org/abs/1508.05326) —— SNLI。
- [Williams, Nangia, Bowman (2017)。通过推理实现句子理解的广泛覆盖挑战语料库](https://arxiv.org/abs/1704.05426) —— MultiNLI。
- [Nie 等人 (2019)。对抗性自然语言推理](https://arxiv.org/abs/1910.14599) —— ANLI 基准测试。
- [Yin, Hay, Roth (2019)。零样本（Zero-shot）文本分类基准测试](https://arxiv.org/abs/1909.00161) —— NLI-as-classifier（将 NLI 作为分类器）。
- [He 等人 (2021)。DeBERTa：具有解耦注意力（Disentangled Attention）的解码增强型 BERT](https://arxiv.org/abs/2006.03654) —— 2026 年 NLI 任务的主力模型。