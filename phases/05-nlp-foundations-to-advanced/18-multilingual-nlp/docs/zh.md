# 多语言自然语言处理 (Multilingual NLP)

> 一个模型，支持 100 多种语言，且其中大多数语言的训练数据为零。跨语言迁移（Cross-lingual transfer）是 2020 年代的一项实用奇迹。

**类型：** 学习
**编程语言：** Python
**前置知识：** 第 5 阶段 · 04（GloVe、FastText、子词/Subword），第 5 阶段 · 11（机器翻译/Machine Translation）
**预计耗时：** 约 45 分钟

## 核心问题

英语拥有数十亿条标注样本，乌尔都语仅有数千条，而迈蒂利语几乎为零。任何面向全球用户的实用自然语言处理（NLP）系统，都必须能够处理那些缺乏特定任务训练数据的长尾语言（Long-tail languages）。

多语言模型（Multilingual models）通过在多种语言上联合训练单一模型来解决这一问题。共享表示（Shared representation）使模型能够将在高资源语言（High-resource languages）中学到的能力迁移至低资源语言（Low-resource languages）。只需在英语情感分析任务上对模型进行微调（Fine-tuning），它便能开箱即用地对乌尔都语进行相当准确的情感预测。这就是零样本跨语言迁移（Zero-shot cross-lingual transfer），它彻底重塑了 NLP 技术向全球交付的方式。

本节将阐明其中的权衡取舍、经典模型（Canonical models），以及一个常让刚接触多语言工作的团队踩坑的关键决策：如何选择用于迁移的源语言（Source language）。

## 核心概念

![通过共享多语言嵌入空间实现跨语言迁移](../assets/multilingual.svg)

**共享词表（Shared vocabulary）。** 多语言模型使用在所有目标语言文本上训练得到的 SentencePiece 或 WordPiece 分词器（Tokenizer）。词表是全局共享的：同一个子词单元（Subword unit）在相关语言中表示相同的语素（Morpheme）。例如，英语和意大利语中的 `anti-` 会被映射为相同的词元（Token）。

**共享表示（Shared representation）。** 在多种语言上进行掩码语言建模（Masked language modeling）预训练的 Transformer 模型会学习到：不同语言中语义相似的句子会产生相似的隐藏状态（Hidden states）。mBERT、XLM-R 和 NLLB 均表现出这一特性。英语中 "cat" 的嵌入向量（Embeddings）会与法语中的 "chat" 和西班牙语中的 "gato" 在向量空间中聚集，完整句子的嵌入向量也是如此。

**零样本迁移（Zero-shot transfer）。** 使用单一语言（通常是英语）的标注数据对模型进行微调。在推理（Inference）阶段，直接将其应用于模型支持的任何其他语言。无需目标语言的标注数据。对于类型学上相近的语言，效果显著；对于差异较大的语言，效果则相对较弱。

**少样本微调（Few-shot fine-tuning）。** 在目标语言中补充 100 到 500 条标注样本。在分类任务上，准确率可跃升至英语基线（Baseline）的 95% 到 98%。这是多语言 NLP 中性价比最高的单一优化手段。

## 主流模型

| 模型 | 年份 | 覆盖语言数 | 备注 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 基于维基百科训练。首个实用的多语言语言模型（Language Model, LM）。在低资源语言（low-resource languages）上表现较弱。 |
| XLM-R | 2019 | 100 种语言 | 基于 CommonCrawl 训练（规模远大于维基百科）。确立了跨语言基线（cross-lingual baseline）。Base 版 2.7 亿参数，Large 版 5.5 亿参数。 |
| XLM-V | 2023 | 100 种语言 | 基于 XLM-R，词表（vocabulary）规模达 100 万 token（对比 25 万）。在低资源语言上表现更优。 |
| mT5 | 2020 | 101 种语言 | 采用 T5 架构，用于多语言生成任务。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 开源的 1760 亿参数大语言模型（Large Language Model, LLM），采用多语言训练。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言大语言模型。在阿拉伯语、印地语和斯瓦希里语上表现强劲。 |

根据具体应用场景进行选择。分类任务通常以 XLM-R-base 作为稳妥的默认选项。生成任务则需根据是机器翻译还是开放文本生成，选择 mT5 或 NLLB。大语言模型类任务建议搭配 Aya-23 或 Claude，并辅以明确的多语言提示（prompting）。

## 源语言选择（2026 年研究）

大多数团队默认使用英语作为微调（fine-tuning）的源语言。但近期研究（2026 年）表明，这种做法往往并不正确。

语言相似度（language similarity）比原始语料库规模更能预测跨语言迁移质量（transfer quality）。对于斯拉夫语系目标语言，德语或俄语的效果通常优于英语。对于印度语系目标语言，印地语通常也优于英语。**qWALS** 相似度指标（2026 年提出，基于《世界语言结构图谱》特征）对此进行了量化。**LANGRANK**（Lin 等人，ACL 2019）则是另一种较早的方法，它综合语言相似度、语料库规模和谱系关系对候选源语言进行排序。

实践建议：如果你的目标语言存在类型学相近的高资源亲属语言，请优先尝试使用该语言进行微调，然后再与英语微调的结果进行对比。

## 动手构建

### Step 1: zero-shot cross-lingual classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

One model, three languages, same API. XLM-R trained on NLI data transfers well to classification via the entailment trick.

### Step 2: multilingual embedding space

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Translations land close in embedding space. A different English sentence lands further. This is what makes cross-lingual retrieval, clustering, and similarity work.

### Step 3: few-shot fine-tuning strategy

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

For 100-500 target-language examples, `num_train_epochs=5` and `learning_rate=2e-5` are the safe defaults. Higher learning rates cause the multilingual alignment to collapse and you get an English-only model.

## 真正有效的评估

- **各语言在独立测试集（held-out sets）上的准确率。** 切勿使用聚合指标。聚合数据会掩盖长尾（long tail）问题。
- **与单语基线模型（monolingual baseline）进行基准对比。** 对于数据量充足的语言，从零开始训练的单语模型有时反而优于多语言模型。务必进行测试验证。
- **实体级测试。** 针对目标语言中的命名实体。多语言模型对于与拉丁字母差异较大的文字系统，其分词（tokenization）能力通常较弱。
- **跨语言一致性（cross-lingual consistency）。** 两种语言中表达相同含义的文本应产生相同的预测结果。请量化测量其差异。

## 实际应用

2026 年技术栈推荐：

| 任务 | 推荐方案 |
|-----|-------------|
| 100 种语言分类 | 微调后的 XLM-R-base (~270M) |
| 零样本（zero-shot）文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入（sentence embeddings） | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 200 种语言翻译 | `facebook/nllb-200-distilled-600M`（参见第 11 课） |
| 多语言生成式模型 | Claude, GPT-4, Aya-23, mT5-XXL |
| 低资源语言自然语言处理（NLP） | XLM-V，或在相关高资源语言上进行领域特定的微调（fine-tune） |

如果性能至关重要，请务必为目标语言的微调（fine-tuning）预留预算。零样本（zero-shot）只是一个起点，而非最终解决方案。

### 分词开销（低资源语言为何表现不佳）

多语言模型在所有语言间共享同一个分词器（tokenizer）。该词表（vocabulary）是在以英语、法语、西班牙语、中文和德语为主导的语料库上训练而成的。对于主导语言集之外的任何语言，三种“开销”会悄然叠加：

- **分词膨胀开销（Fertility tax）。** 低资源语言文本按词切分出的 token 数量远多于英语。一句印地语所需的 token 数量可能是同等含义英语句子的 3-5 倍。这 3-5 倍的膨胀会严重消耗你的上下文窗口（context window）、训练效率并增加延迟。
- **变体恢复开销（Variant recovery tax）。** 每一个拼写错误、变音符号差异、Unicode 规范化不匹配或大小写变化，在嵌入空间（embedding space）中都会变成冷启动的无关序列。模型无法学习到母语者认为理所当然的正字法对应关系。
- **容量溢出开销（Capacity spillover tax）。** 前两种开销会占用上下文位置、网络层深度和嵌入维度。最终留给实际推理的容量，系统性地少于同一模型为高资源语言提供的容量。

实际表现症状：你的模型在印地语上训练过程正常，损失曲线（loss curve）看起来没问题，评估困惑度（perplexity）也合理，但生产环境的输出却存在细微错误。句子中途的形态学（morphology）结构崩溃。罕见的词形变化（inflections）始终无法恢复。**你无法单纯依靠扩大数据规模来弥补分词器本身的缺陷。**

缓解措施：为你的目标语言选择覆盖度良好的分词器（XLM-V 的 100 万 token 词表可直接解决此问题）；在训练前，在预留的目标语言文本上验证分词膨胀率；对于真正的长尾文字系统，使用字节级回退机制（SentencePiece `byte_fallback=True` 或 GPT-2 风格的字节级 BPE），确保永远不会出现词表外（OOV）字符。

## 部署上线

保存为 `outputs/skill-multilingual-picker.md`：

---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语上，针对每种语言运行包含10个句子的零样本分类（Zero-shot Classification）流水线。分别报告各语言的准确率。预期结果应为：法语表现强劲，印地语尚可，阿拉伯语表现存在波动。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型混合语言语料库上构建跨语言检索器（Cross-lingual Retriever）。使用英语进行查询，检索任意语言的文档。测量 Recall@5。
3. **困难。** 针对印地语分类任务，对比以英语为源语言和以印地语为源语言的微调（Fine-tuning）效果。在两种设置下，均使用500个目标语言样本进行少样本微调（Few-shot Fine-tuning）。报告哪种源语言能带来更高的印地语准确率，以及具体高出多少。这本质上是 LANGRANK 理论的微缩实践。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| 多语言模型（Multilingual Model） | 一个模型，多种语言 | 跨语言共享词表与参数。 |
| 跨语言迁移（Cross-lingual Transfer） | 在一种语言上训练，在另一种语言上运行 | 在源语言上微调，在目标语言上评估，且无需目标语言标签。 |
| 零样本（Zero-shot） | 无目标语言标签 | 无需在目标语言上微调即可实现迁移。 |
| 少样本（Few-shot） | 少量目标语言标签 | 使用100-500个目标语言样本进行微调。 |
| mBERT | 首个多语言语言模型（LM） | 基于维基百科预训练的104语言 BERT。 |
| XLM-R | 标准跨语言基线模型 | 基于 CommonCrawl 预训练的100语言 RoBERTa。 |
| NLLB | Meta 的200语言机器翻译（MT）模型 | No Language Left Behind（不让任何语言掉队）。包含55种低资源语言。 |

## 延伸阅读

- [Conneau 等人（2019）。大规模无监督跨语言表示学习](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette（2019）。多语言 BERT 的多语言性究竟如何？](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移（Cross-lingual Transfer）研究方向的奠基性分析论文。
- [Costa-jussà 等人（2022）。不让任何语言掉队](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün 等人（2024）。Aya 模型：一款指令微调（Instruction Fine-tuning）的开放获取多语言语言模型](https://arxiv.org/abs/2402.07827) — Aya，Cohere 推出的多语言大语言模型（Large Language Model, LLM）。
- [语言相似度预测跨语言迁移学习（Cross-lingual Transfer Learning）性能（2026）](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言（Source Language）相关论文。