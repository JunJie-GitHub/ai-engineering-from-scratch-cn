# 问答系统 (Question Answering Systems)

> 三大系统塑造了现代问答 (QA) 技术。抽取式 (Extractive) 定位答案片段。检索增强式 (Retrieval-augmented) 将答案锚定于文档。生成式 (Generative) 直接产出答案。如今的每一个现代 AI 助手都是这三者的融合。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 5 阶段 · 11（机器翻译），第 5 阶段 · 10（注意力机制）
**Time:** 约 75 分钟

## 问题定义

用户输入“第一代 iPhone 是什么时候发布的？”，期望得到的答案是“2007 年 6 月 29 日”。而不是“苹果的历史悠久且丰富多彩”，也不是孤零零的“2007”缺乏上下文。用户需要的是直接、有据可查且正确的答案。

过去十年间，三种架构主导了问答 (QA) 领域的发展。

- **抽取式问答 (Extractive QA)。** 给定一个问题以及已知包含答案的文本段落，找出答案片段在段落中的起始和结束索引。SQuAD 是该领域的标准基准测试。
- **开放域问答 (Open-domain QA)。** 不提供参考段落。首先检索相关段落，然后从中抽取或生成答案。这是当今所有检索增强生成 (RAG) 流水线的基石。
- **生成式/闭卷问答 (Generative / Closed-book QA)。** 大语言模型 (LLM) 直接依靠其参数化记忆 (parametric memory) 进行回答。无需检索步骤。推理速度最快，但事实可靠性最低。

2026 年的趋势是混合架构：先检索出最相关的若干段落，然后提示生成式模型基于这些段落进行回答。这就是检索增强生成 (RAG)，第 14 课将深入讲解检索部分。本课则专注于构建问答部分。

## 核心概念

![问答系统架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式 (Extractive)。** 使用 Transformer 将问题与段落联合编码。训练两个预测头 (heads)，分别预测答案的起始和结束 Token 索引。损失函数为有效位置上的交叉熵 (cross-entropy)。输出为段落中的一个连续片段。其架构设计决定了它绝不会产生幻觉 (hallucination)，但也无法处理段落中未包含答案的问题。

**检索增强式 (Retrieval-augmented, RAG)。** 分为两个阶段。首先，检索器 (retriever) 从语料库 (corpus) 中找出排名最靠前的 `k` 个段落。其次，阅读器 (reader，抽取式或生成式) 利用这些段落生成答案。检索器与阅读器的分离设计使得两者可以独立训练与评估。现代 RAG 架构通常会在两者之间加入重排序器 (reranker)。

**生成式 (Generative)。** 仅解码器架构的大语言模型 (decoder-only LLM，如 GPT、Claude、Llama) 直接基于学习到的权重进行回答。无需检索步骤。在常识问题上表现优异，但在罕见或近期事实面前极易出错。其幻觉率与预训练数据 (pretraining data) 中该事实的出现频率呈负相关。

## 动手构建

### 步骤 1：使用预训练模型进行抽取式问答 (Extractive QA)

from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)

{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}

`deepset/roberta-base-squad2` 模型在 SQuAD 2.0 数据集上进行了训练，该数据集包含无法回答的问题。默认情况下，`question-answering` 流水线 (Pipeline) 即使模型的无答案得分 (Null Score) 最高，也会返回得分最高的文本片段 (Span)——它*不会*自动返回空答案。若要明确启用“无答案”行为，请在调用流水线时传入 `handle_impossible_answer=True`：此时仅当无答案得分超过所有片段得分时，流水线才会返回空答案。无论采用哪种方式，都请务必检查 `score` 字段。

### 步骤 2：检索增强流水线（示例）(Retrieval-Augmented Pipeline)

from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))

两阶段流水线 (Two-stage Pipeline)。稠密检索器 (Dense Retriever)（基于 Sentence-BERT）通过语义相似度 (Semantic Similarity) 查找相关段落。抽取式阅读器 (Extractive Reader)（基于 RoBERTa-SQuAD）从合并后的 Top 段落中提取答案片段。该方案适用于小型语料库 (Corpus)。对于百万级文档的语料库，请使用 FAISS 或向量数据库 (Vector Database)。

### 步骤 3：基于 RAG 的生成式问答 (Generative with RAG)

def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)

提示词模式 (Prompt Pattern) 的设计至关重要。与基础提示 (Naive Prompting) 相比，明确要求模型基于上下文进行回答，并在上下文信息不足时返回“我不知道”，可将幻觉率 (Hallucination Rate) 降低 40% 至 60%。更复杂的模式还会加入引用来源、置信度评分以及结构化提取。

### 步骤 4：贴近真实场景的评估 (Evaluation)

SQuAD 数据集采用**精确匹配 (Exact Match, EM)** 和**词元级 F1 分数 (Token-level F1)**。EM 是在标准化处理 (Normalization)（转为小写、去除标点符号、移除冠词）后的严格匹配——预测结果必须完全一致，否则得分为 0。F1 分数则基于预测结果与参考答案之间的词元 (Token) 重叠度计算，允许部分得分。这两种指标对同义改写 (Paraphrase) 的评分都偏低：例如“June 29, 2007”与“June 29th, 2007”通常 EM 得分为 0（序数词破坏了标准化规则），但由于词元重叠，仍能获得较高的 F1 分数。

面向生产环境的问答系统评估：

- **答案准确性 (Answer Accuracy)**（由大语言模型或人工评判，因为传统指标无法捕捉语义等价性）。
- **引用准确性 (Citation Accuracy)**。引用的段落是否真正支撑了答案？通过将生成的引用与检索到的段落进行字符串匹配，即可轻松实现自动检查。
- **拒答校准 (Refusal Calibration)**。当答案不在检索到的段落中时，系统能否正确回答“我不知道”？需测量错误自信率 (False Confidence Rate)。
- **检索召回率 (Retrieval Recall)**。在评估阅读器之前，先衡量检索器是否将正确的段落排进了 Top-`k`。如果段落缺失，阅读器也无法凭空生成答案。

### RAGAS：2026 年生产级评估框架 (RAGAS)

`RAGAS` 专为 RAG 系统设计，是 2026 年默认交付的评估工具。它无需黄金标准答案 (Gold References) 即可对四个维度进行评分：

- **忠实度 (Faithfulness)**。答案中的每一项主张是否均源自检索到的上下文？通过基于自然语言推理的蕴含关系 (NLI-based Entailment) 进行测量。这是衡量幻觉的核心指标。
- **答案相关性 (Answer Relevance)**。答案是否切中问题？通过从答案生成假设性问题，并将其与真实问题进行比对来测量。
- **上下文精确率 (Context Precision)**。在检索到的文本块 (Chunks) 中，实际相关的比例是多少？精确率低意味着提示词中混入了噪声。
- **上下文召回率 (Context Recall)**。检索到的集合是否包含了所有必要信息？召回率低意味着阅读器无法成功作答。

免参考评分 (Reference-free Scoring) 允许你在无需人工整理标准答案的情况下，直接基于线上生产流量进行评估。对于精确匹配指标失效的开放式问题，可在此基础上叠加“大语言模型即裁判” (LLM-as-Judge) 机制。

执行 `pip install ragas`。接入你的检索器与阅读器。每次查询即可获取四个标量指标。针对性能回退 (Regressions) 设置告警。

## 实际应用

2026 年技术栈。

| 使用场景 | 推荐方案 |
|---------|-------------|
| 给定段落，定位答案片段 | `deepset/roberta-base-squad2` |
| 针对固定语料库，不可接受闭卷（closed-book）生成 | 检索增强生成（Retrieval-Augmented Generation, RAG）：稠密检索器（dense retriever） + 大语言模型（Large Language Model, LLM）阅读器 |
| 针对文档库的实时查询 | RAG 结合混合检索器（BM25 + 稠密） + 重排器（reranker）（见第 14 课） |
| 对话式问答（多轮追问） | 结合对话历史的 LLM + 每轮触发 RAG |
| 高事实性、强监管领域 | 抽取式问答（Extractive QA）基于权威语料库；绝不单独使用生成式（generative）模型 |

抽取式问答在 2026 年已不再流行，因为结合大语言模型的检索增强生成能够处理更多场景。但在需要逐字引用的场景中，它依然被广泛部署：例如法律研究、合规审查与审计工具。

## 部署交付

保存为 `outputs/skill-qa-architect.md`：

---
name: 问答架构师
description: 选择问答架构、检索策略与评估方案。
version: 1.0.0
phase: 5
lesson: 13
tags: [自然语言处理, 问答, 检索增强生成]
---

根据需求（语料库规模、问题类型、事实性约束、延迟预算），输出以下内容：

1. 架构。抽取式、结合抽取式阅读器的 RAG、结合生成式阅读器的 RAG，或闭卷大语言模型。附一句理由。
2. 检索器。无、BM25、稠密（注明编码器名称）或混合检索。
3. 阅读器。SQuAD 微调模型、指定名称的大语言模型，或“领域微调的 DistilBERT”。
4. 评估。抽取式基准测试使用精确匹配（Exact Match, EM） + F1 分数；生产环境使用答案准确率 + 引用准确率 + 拒绝校准（refusal calibration）。明确说明测量指标及测量方法。

对于监管或合规敏感类问题，拒绝使用闭卷大语言模型作答。拒绝任何缺乏检索召回率基线（retrieval-recall baseline）的问答系统（若不知检索器是否召回了正确段落，则无法评估阅读器）。将需要多跳推理（multi-hop reasoning）的问题标记为需使用专门的多跳检索器（如基于 HotpotQA 训练的系统）。

## 练习

1. **简单。** 在 10 篇维基百科段落上搭建上述 SQuAD 抽取式流水线。手动编写 10 个问题。统计答案正确的频率。若段落与问题质量较高，正确数应在 7-9 个之间。
2. **中等。** 添加拒绝分类器（refusal classifier）。当最高检索得分低于阈值（例如余弦相似度 0.3）时，返回“我不知道”，而非调用阅读器。在预留验证集上调整该阈值。
3. **困难。** 在你自选的 10,000 篇文档语料库上构建 RAG 流水线。实现结合 RRF 融合（Reciprocal Rank Fusion, RRF）（见第 14 课）的混合检索（BM25 + 稠密）。对比有无混合检索步骤时的答案准确率。记录哪些类型的问题受益最大。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式问答 (Extractive QA) | 定位答案片段 | 预测给定段落中答案的起始与结束索引位置。 |
| 开放域问答 (Open-domain QA) | 基于语料库的问答 | 不提供指定段落；需先执行检索再进行作答。 |
| 检索增强生成 (RAG) | 先检索后生成 | 检索增强生成技术。采用“检索器 + 阅读器”流水线架构。 |
| SQuAD | 标准基准 | 斯坦福问答数据集 (Stanford Question Answering Dataset)。采用精确匹配 (Exact Match, EM) 与 F1 分数作为评估指标。 |
| 幻觉 (Hallucination) | 捏造答案 | 阅读器生成的输出缺乏检索到的上下文支持。 |
| 拒答校准 (Refusal Calibration) | 懂得何时保持沉默 | 系统在无法作答时能正确输出“我不知道”。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) —— 问答领域的基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) —— DPR，问答任务中标准的稠密检索器 (Dense Retriever)。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) —— 正式提出并命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) —— 关于 RAG 技术的全面综述。