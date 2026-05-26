# 大语言模型 (LLM) 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配 (Exact Match) 与 F1 分数无法衡量语义等价性。人工审核难以实现规模化。将大语言模型作为裁判 (LLM-as-judge) 是生产环境的标准答案——只要经过充分校准，即可信任其评估结果。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 13（问答系统），第 5 阶段 · 14（信息检索）
**耗时：** 约 75 分钟

## 核心问题

你的检索增强生成 (RAG) 系统回答：“2007年6月29日。”标准参考答案 (gold reference) 是：“2007年6月29日。”精确匹配 (Exact Match) 得分为 0，F1 分数约为 75%。而人类评估会给出 100% 的分数。

现在将测试用例数量乘以 10,000。再乘以检索器 (retriever)、文本分块 (chunking)、提示词 (prompt) 或模型每一次的迭代变更。你需要一个能够理解语义、可低成本规模化运行、不会在性能回退 (regressions) 时给出虚假报告，并能准确暴露关键失败模式 (failure modes) 的评估器。

2026 年，三大框架已成为该领域的主导方案。

- **RAGAS。** 检索增强生成评估 (Retrieval-Augmented Generation ASsessment)。提供四项 RAG 核心指标（忠实度 (faithfulness)、答案相关性 (answer-relevance)、上下文精确度 (context-precision)、上下文召回率 (context-recall)），底层支持自然语言推理 (NLI) 与大语言模型裁判 (LLM-judge)。具备学术研究支撑，架构轻量。
- **DeepEval。** 专为大语言模型设计的 Pytest 框架。内置 G-Eval、任务完成度 (task-completion)、幻觉 (hallucination) 及偏见 (bias) 等评估指标。原生支持 CI/CD 流水线。
- **G-Eval。** 一种评估方法（同时也是 DeepEval 的一项指标）：基于思维链 (chain-of-thought) 的大语言模型裁判机制，支持自定义评估标准，输出 0-1 的标准化分数。

这三者均依赖于大语言模型裁判 (LLM-as-judge) 机制。本课程将帮助你建立对该方法的直观理解，并掌握围绕其构建的信任层 (trust layer)。

## 核心概念

![四个评估维度，大语言模型裁判（LLM-as-judge）架构](../assets/llm-evaluation.svg)

**大语言模型裁判（LLM-as-judge）。** 使用大语言模型替代静态指标，根据既定评分标准对模型输出进行打分。给定 `(query, context, answer)`，向裁判 LLM 发送提示词：“在忠实度（Faithfulness）上打 0-1 分。”并返回该分数。

**为何有效：** LLM 能够以极低的成本逼近人类判断水平。以 GPT-4o-mini 为例，其单次评分成本仅约 0.003 美元，使得运行包含 1000 个样本的回归评估（Regression Evaluation）总成本可控制在 5 美元以内。

**为何会静默失败：**

1. **裁判偏差（Judge Bias）。** 裁判模型倾向于偏好更长的回答、同系列模型生成的回答，以及与提示词风格匹配的回答。
2. **JSON 解析失败（JSON Parsing Failures）。** 格式错误的 JSON → 产生 NaN 分数 → 在聚合计算中被静默剔除。RAGAS 用户对此深有体会。需使用 `try/except` 进行拦截，并设置明确的失败处理模式。
3. **模型版本漂移（Model Version Drift）。** 升级裁判模型会导致所有指标发生变化。务必固定裁判模型的名称与版本号。

**RAG 四大核心指标。**

| 指标 | 核心问题 | 底层实现 |
|--------|----------|---------|
| 忠实度（Faithfulness） | 答案中的每一项主张是否均源自检索到的上下文？ | 基于自然语言推理（NLI）的蕴含判断 |
| 答案相关性（Answer Relevance） | 答案是否切实回应了问题？ | 从答案生成假设性问题，并与真实问题比对 |
| 上下文精确率（Context Precision） | 在检索到的文本块中，有多大比例是相关的？ | LLM 裁判评估 |
| 上下文召回率（Context Recall） | 检索是否返回了所需的全部内容？ | 基于标准答案（Gold Answer）的 LLM 裁判评估 |

**G-Eval。** 定义自定义评估标准：“答案是否引用了正确的来源？”该框架会自动将其扩展为思维链（Chain-of-Thought）评估步骤，随后给出 0-1 的评分。非常适合用于评估 RAGAS 未涵盖的特定领域质量维度。

**评分校准（Calibration）。** 在未与人工标注结果进行相关性验证之前，切勿盲目信任裁判模型的原始评分。选取 100 个手工标注的样本进行测试。绘制裁判评分与人工评分的对比图。计算斯皮尔曼相关系数（Spearman Rho）。若 rho < 0.7，则说明你的裁判评分标准仍需优化。

## 动手构建

### 步骤 1：基于自然语言推理（Natural Language Inference, NLI）的忠实度（Faithfulness）评估（RAGAS 风格）

from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)

将回答拆解为原子级声明（atomic claims）。利用 NLI 模型逐一验证每个声明与检索上下文（context）的一致性。忠实度 = 获得支持的声明所占比例。

### 步骤 2：回答相关性（Answer Relevance）

import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)

若回答所隐含的问题与实际提问存在偏差，相关性得分将会降低。

### 步骤 3：G-Eval 自定义评估指标

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)

这些评估步骤构成了评分细则（rubric）。相较于隐式的“给出 0-1 分”提示词，明确的步骤能带来更稳定的评估结果。

### 步骤 4：持续集成门禁（CI Gate）

import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"

将其封装为 pytest 测试文件。在每次拉取请求（Pull Request, PR）时自动运行。若检测到指标回退（regressions），则拦截合并操作。

### 步骤 5：从零构建简易评估脚本

详见 `code/main.py`。该脚本仅使用 Python 标准库，实现了忠实度（回答声明与上下文的重合度）与相关性（回答词元（tokens）与问题词元的重合度）的近似计算。此代码不适用于生产环境，仅用于展示评估流程的基本形态。

## 潜在陷阱

- **缺乏校准（Calibration）。** 与人工标注相关性仅为 0.3 的评判模型（Judge）等同于噪声。在部署前必须进行校准运行。
- **自我评估（Self-evaluation）。** 使用同一个大语言模型（LLM）既生成内容又进行评判，会导致分数虚高 10-20%。评判模型应选用不同的模型系列。
- **成对评判中的位置偏差（Positional bias）。** 评判模型倾向于选择首先呈现的选项。务必随机打乱顺序并双向运行测试。
- **原始聚合分数掩盖失败案例。** 0.85 的平均分往往掩盖了 5% 的灾难性失败。务必检查低分位（Bottom quantile）数据。
- **黄金数据集退化（Golden dataset rot）。** 未进行版本控制且随时间发生漂移的评估集（Eval set）会破坏纵向对比。每次变更都需为数据集打标签。
- **大语言模型成本。** 在规模化应用中，评判模型的调用将占据主要成本。应选用满足校准阈值的最廉价模型，例如 GPT-4o-mini、Claude Haiku 或 Mistral-small。

## 使用场景

2026 年技术栈：

| 使用场景 | 框架 |
|---------|-----------|
| 检索增强生成（RAG）质量监控 | RAGAS（4 项指标） |
| 持续集成/持续部署（CI/CD）回归门禁 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 内置的 G-Eval |
| 线上实时流量监控 | RAGAS（无参考模式） |
| 人机协同（Human-in-the-loop）抽检 | LangSmith 或 Phoenix（带标注界面） |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型技术栈：使用 RAGAS 进行监控，DeepEval 用于持续集成（CI），G-Eval 用于探索新维度。建议同时运行这三者，它们之间的分歧往往能提供有价值的参考。

## 交付指南

保存为 `outputs/skill-eval-architect.md`：

---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.

## 练习

1. **简单。** 在 10 个已知存在幻觉（Hallucination）的 RAG 示例上使用 RAGAS。验证忠实度（Faithfulness）指标是否能准确捕捉到每一个幻觉。
2. **中等。** 人工标注 50 个问答（QA）答案的正确性（0-1 分）。使用 G-Eval 进行评分。计算评判模型与人工标注之间的斯皮尔曼等级相关系数（Spearman rho）。
3. **困难。** 使用 DeepEval 构建 pytest 持续集成（CI）门禁。故意使检索器（Retriever）性能回退。验证门禁是否成功拦截。通过对最低 10% 的数据进行阈值检查，添加低分位告警功能。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 大模型裁判 (LLM-as-judge) | 使用大语言模型打分 | 根据评分标准提示评判模型，对输出进行 0-1 分打分。 |
| RAGAS | RAG 指标库 | 包含 4 项无参考检索增强生成 (RAG) 指标的开源评估框架。 |
| 忠实度 (Faithfulness) | 答案是否有据可依？ | 答案中的陈述被检索上下文所蕴含的比例。 |
| 上下文精确率 (Context Precision) | 检索到的文本块是否相关？ | 实际起作用的 Top-K 文本块所占比例。 |
| 上下文召回率 (Context Recall) | 检索是否找全了？ | 被检索文本块所支持的标准答案陈述的比例。 |
| G-Eval | 自定义大模型评判器 | 结合评分标准、思维链 (Chain-of-Thought) 评估步骤与 0-1 打分。 |
| 校准 (Calibration) | 信任但需验证 | 评判模型打分与人工打分之间的斯皮尔曼等级相关系数 (Spearman Correlation)。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 面向生产的开源技术栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 探讨偏差、校准与局限性。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval 与 Phoenix 的统一框架。