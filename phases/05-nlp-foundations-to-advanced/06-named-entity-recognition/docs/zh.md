# 命名实体识别 (Named Entity Recognition)

> 提取出名称。听起来很简单，直到你面对边界模糊、实体嵌套以及领域专有名词时才会发现其中的难度。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 02（词袋模型 (Bag of Words, BoW) + TF-IDF），第 5 阶段 · 03（词嵌入 (Word Embeddings)）
**时长：** 约 75 分钟

## 问题

“Apple 在美国因 iPhone 搜索协议起诉 Google。”包含五个实体：Apple（组织机构 ORG）、Google（组织机构 ORG）、iPhone（产品 PRODUCT）、search deal（可能）、US（地缘政治实体 GPE）。一个优秀的命名实体识别 (NER) 系统能准确提取所有实体并赋予正确类型。而糟糕的系统会漏掉 iPhone，将水果“苹果”与公司“Apple”混淆，并将“US”错误标记为人名 (PERSON)。

NER 是支撑所有结构化信息提取流水线的核心组件。无论是简历解析、合规日志扫描、医疗记录脱敏、搜索查询理解、聊天机器人回复的事实依据 (Grounding)，还是法律合同提取，都离不开它。你往往察觉不到它的存在，却始终依赖它。

本课程将带你从经典方法（基于规则 (Rule-based)、隐马尔可夫模型 (Hidden Markov Model, HMM)、条件随机场 (Conditional Random Field, CRF)）走向现代方法（双向长短期记忆网络-条件随机场 (BiLSTM-CRF)，再到 Transformer）。每一步都旨在解决前一步的特定局限性。这种演进模式本身就是本课的核心。

## 核心概念

**BIO 标注法 (BIO Tagging)**（或 BILOU）将实体提取转化为序列标注 (Sequence Labeling) 问题。为每个词元 (Token) 标注 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（非实体）。

Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O

多词元实体通过链式连接：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 格式的模型能够提取任意长度的文本片段 (Span)。

架构演进路线：

- **基于规则 (Rule-based)。** 正则表达式 + 专有名词表 (Gazetteer) 查找。对已知实体精度高，但对新实体覆盖率为零。
- **隐马尔可夫模型 (Hidden Markov Model, HMM)。** 计算给定标签下词元的发射概率 (Emission Probability)，以及标签间的转移概率 (Transition Probability)。使用维特比解码 (Viterbi Decoding)。需在标注数据上训练。
- **条件随机场 (Conditional Random Field, CRF)。** 类似 HMM 但属于判别式 (Discriminative) 模型，因此可以融合任意特征（词形、大小写、相邻词等）。在 2026 年，它仍是低资源部署场景下经典的工业级主力模型。
- **BiLSTM-CRF。** 使用神经网络特征替代人工设计特征。双向长短期记忆网络 (LSTM) 从正反两个方向读取句子，顶层的 CRF 层用于强制保证标签序列的一致性。
- **基于 Transformer 的架构。** 在 BERT 基础上微调并添加词元分类头 (Token Classification Head)。精度最高，但计算开销最大。

## 动手实践

### 步骤 1：BIO 标注（BIO Tagging）辅助函数

def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans

>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]

### 步骤 2：手工特征（Hand-crafted Features）

对于传统的（非神经网络的）命名实体识别（Named Entity Recognition, NER）而言，特征设计至关重要。以下是一些实用的特征：

def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对于识别专有名词具有很高的区分度。

### 步骤 3：基于规则与 gazetteer（专有名词词典）的简单基线模型

ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels

生产环境中的 gazetteer（专有名词词典）通常包含从维基百科和 DBpedia 抓取的上百万条词条。其覆盖率很高，但消歧能力极差（例如无法区分作为公司的 `Apple` 和作为水果的苹果）。这正是统计模型最终胜出的原因。

### 步骤 4：条件随机场（Conditional Random Field, CRF）步骤（示意，非完整实现）

在缺乏概率论基础的情况下，仅用 50 行代码从零实现完整的 CRF 并无太大启发意义。建议直接使用 `sklearn-crfsuite`：

import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)

`c1` 和 `c2` 分别对应 L1 和 L2 正则化（Regularization）。设置 `all_possible_transitions=True` 可使模型学习到非法序列（例如在 `O` 之后直接出现 `I-ORG`）的概率极低。这正是 CRF 无需手动编写约束规则即可强制保证 BIO 标注一致性的原理。

### 步骤 5：BiLSTM-CRF 带来的改进

特征转为自动学习。输入为词元嵌入（Token Embeddings，如 GloVe 或 fastText）。双向长短期记忆网络（Bidirectional Long Short-Term Memory, BiLSTM）分别从左到右和从右到左读取序列。拼接后的隐藏状态会输入到 CRF 输出层。CRF 依然负责保证标签序列的一致性，而 LSTM 则用学习到的特征替代了手工设计的特征。

import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions

对于 CRF 层，可使用 `torchcrf.CRF`（通过 `pip install pytorch-crf` 安装）。相较于手工特征 CRF，其性能提升是可衡量的，但除非你拥有数万条标注句子，否则提升幅度可能低于预期。

## Use It

spaCy 开箱即用地提供了生产级命名实体识别（Named Entity Recognition, NER）功能。

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")

Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE

注意 `iPhone` 被标记为 `ORG` 而非 `PRODUCT` —— spaCy 的小型模型在产品实体覆盖方面较弱。大型模型（`en_core_web_lg`）表现更好，而基于 Transformer 的模型（`en_core_web_trf`）则更胜一筹。

使用 Hugging Face 进行基于 BERT 的 NER：

from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))

[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]

参数 `aggregation_strategy="simple"` 会将连续的 B-X、I-X 词元（token）合并为一个跨度（span）。若不设置该参数，你将获得词元级别的标签，并需要自行进行合并。

### 基于大语言模型（Large Language Model, LLM）的 NER（2026 年方案）

零样本（Zero-shot）和少样本（Few-shot）的 LLM NER 目前在许多领域已能与微调模型相媲美，且在标注数据稀缺时表现显著更优。

- **零样本提示（Zero-shot prompting）。** 向 LLM 提供实体类型列表和示例模式（schema），并要求其输出 JSON 格式。该方法开箱即用；在全新领域上的准确率处于中等水平。
- **ZeroTuneBio 风格提示。** 将任务拆解为候选提取 → 含义解释 → 判断 → 复核。多阶段提示（而非单步提示）能显著提升生物医学 NER 的准确率。该模式同样适用于法律、金融和科学领域。
- **结合检索增强生成（Retrieval-Augmented Generation, RAG）的动态提示。** 在每次推理调用时，从少量标注种子集中检索最相似的标注示例，并动态构建少样本提示。在 2026 年的基准测试中，该方法使 GPT-4 在生物医学 NER 上的 F1 分数较静态提示提升了 11-12%。
- **按实体类型拆解。** 对于长文档，单次调用提取所有实体类型会导致召回率随文档长度增加而下降。改为针对每种实体类型分别执行一次提取流程。虽然推理成本更高，但准确率会大幅提升。这是处理临床病历和法律合同的标准模式。

截至 2026 年的生产环境建议：在收集训练数据之前，先以 LLM 零样本作为基线。通常其 F1 分数已足够好，以至于你根本无需进行微调。

### 传统 NER 仍具优势的领域

即使 LLM 已触手可及，传统 NER 在以下场景仍具优势：

- 延迟预算低于 50 毫秒。
- 你拥有数千条标注样本，且需要 98% 以上的 F1 分数。
- 领域本体（ontology）稳定，预训练的条件随机场（Conditional Random Field, CRF）或双向长短期记忆网络（Bidirectional Long Short-Term Memory, BiLSTM）能够良好迁移。
- 监管要求必须使用本地部署（on-prem）的非生成式模型。

### 传统 NER 的局限性

- **领域偏移（Domain shift）。** 在 CoNLL 数据集上训练的 NER 模型处理法律合同时，表现甚至不如 gazetteer 词典（gazetteer）。请在你的目标领域上进行微调。
- **嵌套实体（Nested entities）。** “美国银行大厦（Bank of America Tower）”同时属于 `ORG` 和 `FACILITY`。标准的 BIO 标注体系（BIO）无法表示重叠的文本片段。你需要使用嵌套 NER（多轮提取或基于跨度的模型）。
- **长实体。** “美国联邦存款保险公司（United States Federal Deposit Insurance Corporation）”。基于词元级别的模型有时会将其错误切分。请使用 `aggregation_strategy` 参数或进行后处理。
- **稀疏类型（Sparse types）。** 医疗 NER 标签如 `DRUG_BRAND`、`ADVERSE_EVENT`、`DOSE`。通用模型对此一无所知。在此类场景下，`Scispacy` 和 `BioBERT` 是理想的起点。

## 交付上线

保存为 `outputs/skill-ner-picker.md`：

---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在 10 个句子上验证往返转换的一致性。
2. **中等。** 使用上述的 `sklearn-crfsuite` 条件随机场（CRF）在 CoNLL-2003 英文命名实体识别（NER）数据集上进行训练。使用 `seqeval` 报告每个实体的 F1 分数。典型结果：F1 约为 84。
3. **困难。** 在特定领域的 NER 数据集（医疗、法律或金融）上微调 `distilbert-base-cased`。与 `spaCy` 小型模型进行对比。记录数据泄露检查过程，并撰写令你感到意外的发现。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 命名实体识别（NER） | 提取名称 | 为词元跨度标注类型（如 PERSON、ORG、GPE、DATE 等）。 |
| BIO | 标注方案 | `B-X` 表示开始，`I-X` 表示继续，`O` 表示外部。 |
| BILOU | 改进版 BIO | 增加 `L-X`（末尾）和 `U-X`（独立单元），使边界划分更清晰。 |
| 条件随机场（CRF） | 结构化分类器 | 对标签之间的转移进行建模，而不仅仅是发射概率。强制保证序列的有效性。 |
| 嵌套命名实体识别（Nested NER） | 重叠实体 | 一个跨度与其子跨度属于不同的实体。BIO 方案无法表达这种情况。 |
| 实体级 F1（Entity-level F1） | 标准的 NER 评估指标 | 预测的跨度必须与真实跨度完全匹配。词元级（token-level）F1 会高估准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 的奠基性论文。权威之作。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入了现已成为标准的词元分类（token-classification）范式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — 关于 `Doc.ents` 和 `Span` 所有属性的实用参考文档。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的评估指标库。请始终使用它。