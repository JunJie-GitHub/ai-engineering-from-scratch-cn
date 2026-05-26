# 长上下文评估（Long-Context Evaluation）— NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称支持 1000 万（10M）token 的上下文窗口（Context Window）。但在处理 100 万（1M）token 时，8 针 MRCR 测试的得分骤降至 26.3%。宣称容量 ≠ 实际可用容量。长上下文评估能揭示你实际部署模型的真实能力。

**类型：** 学习
**语言：** Python
**前置知识：** 第 5 阶段 · 13（问答），第 5 阶段 · 23（分块策略）
**预计耗时：** 约 60 分钟

## 核心问题

你手头有一份 200 页的合同。模型宣称支持 100 万（1M）token 的上下文。你将合同全文粘贴进去并提问：“终止条款是什么？”模型确实给出了回答——但答案却提取自封面页，因为真正的终止条款位于第 12 万（120k）token 处，已经超出了模型注意力机制（Attention Mechanism）实际能够有效覆盖的范围。

这就是 2026 年面临的上下文容量差距。规格参数表上写着 100 万或 1000 万 token。但现实情况是，其中只有 60% 到 70% 是真正可用的，且“可用性”高度依赖于具体任务。

- **检索（单针大海捞针测试 / Single Needle in Haystack）：** 在前沿模型上，直到宣称的最大长度，表现几乎完美。
- **多跳推理与信息聚合（Multi-hop / Aggregation）：** 在大多数模型中，超过约 12.8 万（128k）token 后性能急剧下降。
- **跨分散事实推理（Reasoning over Dispersed Facts）：** 最先失效的任务类型。

长上下文评估正是为了量化这些维度。本课程将介绍相关基准测试（Benchmarks），阐明它们各自实际测量的指标，并指导你如何为特定业务领域构建自定义的针测试（Needle Test）。

## 核心概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**大海捞针（Needle-in-a-Haystack, NIAH, 2023）。** 在长上下文的指定深度位置插入一个事实（例如“魔法词是 pineapple”），并要求模型将其检索出来。通过遍历深度与上下文长度（context length）的组合进行测试。这是最初的长上下文基准测试。目前前沿模型在此任务上的表现已趋于饱和；因此它是一项必要但不充分的基线。

**RULER（Nvidia, 2024）。** 涵盖 4 大类共 13 种任务类型：检索（单键/多键/多值）、多跳追踪（multi-hop tracing，变量跟踪）、聚合（aggregation，常见词频统计）以及问答（QA）。支持配置上下文长度（4k 至 128k+）。该基准能够有效识别出在 NIAH 上表现饱和但在多跳任务中失败的模型。在 2024 年的测试中，17 款宣称支持 32k+ 上下文的模型里，仅有一半能在 32k 长度下保持原有质量。

**LongBench v2（2024）。** 包含 503 道选择题，上下文长度覆盖 8k 至 200 万词，分为六大任务类别：单文档问答、多文档问答、长上下文内学习（in-context learning）、长对话、代码仓库以及长结构化数据。这是用于评估真实场景下长上下文表现的工业级基准测试。

**MRCR（多轮指代消解，Multi-Round Coreference Resolution）。** 面向大规模的多轮指代消解任务。提供 8 针、24 针和 100 针等不同变体。用于揭示模型在注意力机制（attention）退化前能够同时处理多少事实信息。

**NoLiMa。** “非词汇型针（Non-lexical needle）”。目标信息与查询之间不存在字面重叠；检索需要模型执行一步语义推理。难度高于 NIAH。

**HELMET。** 拼接多篇文档，并针对其中任意一篇提出问题。用于测试模型的选择性注意力（selective attention）。

**BABILong。** 将 bAbI 推理链嵌入到无关的干扰文本中。该基准测试的是“干扰环境中的推理能力（reasoning-in-a-haystack）”，而非单纯的检索能力。

### 实际应报告的指标

- **标称上下文窗口（advertised context window）。** 规格说明书上的数值。
- **有效检索长度（effective retrieval length）。** 在特定阈值（如 90%）下通过 NIAH 测试的上下文长度。
- **有效推理长度（effective reasoning length）。** 在相同阈值下通过多跳或聚合测试的上下文长度。
- **性能衰减曲线（degradation curve）。** 按任务类型绘制的准确率随上下文长度变化的曲线。

规格表上只需保留两个核心数值：有效检索长度与有效推理长度。通常情况下，有效推理长度仅为标称上下文窗口的 25% 至 50%。

## 动手构建

### 步骤 1：为你的领域定制大海捞针测试（Needle In A Haystack, NIAH）

参见 `code/main.py`。其骨架代码如下：

def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0

遍历 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图（heatmap）。该结果即为你目标模型的大海捞针测试（NIAH）性能卡片。

### 步骤 2：多针（multi-needle）变体

def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)

类似“三个魔法词是什么？”的问题要求模型检索出全部三个目标。单针（single-needle）测试的成功率无法预测多针（multi-needle）测试的表现。

### 步骤 3：多跳变量追踪（multi-hop variable tracing）（RULER 风格）

haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"

得出答案需要串联三次赋值操作。在 128k 上下文窗口下，前沿模型（frontier models）在此类任务上的准确率通常会降至 50%~70%。

### 步骤 4：在你的技术栈（stack）上运行 LongBench v2

from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)

报告每个类别的准确率。汇总得分会掩盖任务层面的显著差异。

## 常见陷阱（pitfalls）

- **仅依赖 NIAH（Needle In A Haystack）评估。** 在 100 万 token 上通过 NIAH 测试并不能说明模型具备多跳（multi-hop）推理能力。务必运行 RULER 基准测试或自定义的多跳测试。
- **均匀深度（depth）采样。** 许多实现仅测试 depth=0.5。请测试 depth=0、0.25、0.5、0.75、1.0 —— “中间迷失（lost in the middle）”效应是真实存在的。
- **与填充文本（filler）存在词汇重叠。** 如果“针”（needle）与填充文本共享关键词，检索任务将变得过于简单。请使用 NoLiMa 风格的无重叠“针”。
- **忽略延迟。** 100 万 token 的提示词预填充（prefill）需要 30-120 秒。在评估准确率的同时，务必测量首字延迟（time-to-first-token）。
- **厂商自报数据。** OpenAI、Google 和 Anthropic 都会发布自己的测试分数。务必在你的实际用例上独立重新运行测试。

## 使用指南

2026 年技术栈建议：

| 场景 | 基准测试 |
|-----------|-----------|
| 快速验证 | 3 种深度 × 3 种长度的自定义 NIAH |
| 生产环境模型选型 | 目标长度下的 RULER（13 项任务） |
| 真实场景问答质量 | LongBench v2 单文档问答子集 |
| 多跳推理 | BABILong 或自定义变量追踪（variable-tracing） |
| 对话/多轮交互 | 目标长度下的 MRCR 8 针测试 |
| 模型升级回归测试 | 固定的内部 NIAH + RULER 测试框架，在每个新模型上运行 |

生产环境经验法则：在目标长度下完成 NIAH 测试 + 至少 1 项推理任务之前，切勿轻信模型宣称的上下文窗口（context window）能力。

## 交付与部署

保存为 `outputs/skill-long-context-eval.md`：

---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.

## 练习

1. **简单。** 构建一个包含 3 种深度（0.25、0.5、0.75）× 3 种长度（1k、4k、16k）的 NIAH 测试。在任意模型上运行，并将通过率绘制为 3×3 热力图。
2. **中等。** 增加 3 针变体。测量在每个长度下全部 3 根“针”的检索情况。与相同长度下的单针通过率进行对比。
3. **困难。** 构建一个变量追踪任务（X1 → X2 → X3，共 3 跳），并将其嵌入 64k 的填充文本中。在 3 个前沿模型（frontier models）上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| NIAH (Needle in a Haystack) | 大海捞针测试 | 在填充文本中植入特定事实，要求模型将其检索出来。 |
| RULER | NIAH 的强化版 | 涵盖检索、多跳推理 (Multi-hop)、信息聚合与问答等 13 种任务类型的基准测试。 |
| 有效上下文 (Effective Context) | 真实容量 | 模型准确率仍能维持在设定阈值之上的最大上下文长度。 |
| 中间迷失现象 (Lost in the Middle) | 深度偏差 (Depth Bias) | 模型对长输入文本中间部分的内容关注度显著下降。 |
| 多针测试 (Multi-needle) | 同时处理多个事实 | 植入多个目标事实；主要测试模型注意力机制的多目标调度能力，而非单一检索能力。 |
| MRCR (Multi-round Coreference Resolution) | 多轮指代消解 | 包含 8、24 或 100 个目标的指代消解任务；用于暴露模型的注意力饱和 (Attention Saturation) 问题。 |
| NoLiMa (Non-lexical Needle) | 非词法针 | 目标事实与查询语句不共享任何字面词元 (Token)；要求模型具备深层推理能力。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — NIAH 测试的原始代码仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务长上下文基准测试。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 面向真实应用场景的长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 难度更高的非词法针测试。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 长文本中的复杂推理测试。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 揭示模型深度偏差现象的奠基性论文。