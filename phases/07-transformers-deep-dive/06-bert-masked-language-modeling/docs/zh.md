# BERT — 掩码语言建模 (Masked Language Modeling)

> GPT 预测下一个词，而 BERT 预测缺失的词。仅一句话的差异，却奠定了过去五年所有以嵌入 (Embedding) 为核心的技术演进。

**类型：** 构建实践
**编程语言：** Python
**前置知识：** 第 7 阶段 · 05（完整 Transformer），第 5 阶段 · 02（文本表示 (Text Representation)）
**预计耗时：** 约 45 分钟

## 问题背景

2018 年，每一项自然语言处理 (Natural Language Processing, NLP) 任务——包括情感分析 (Sentiment Analysis)、命名实体识别 (Named Entity Recognition, NER)、问答 (Question Answering, QA) 和文本蕴含 (Textual Entailment)——都需要在各自的标注数据上从零开始训练专属模型。当时并不存在一个预训练好的“理解英语”检查点 (Checkpoint) 供你进行微调 (Fine-tuning)。ELMo (2018) 证明了可以使用双向长短期记忆网络 (Bidirectional LSTM) 预训练上下文嵌入 (Contextual Embeddings)；这确实带来了性能提升，但泛化能力依然有限。

BERT (Devlin 等人, 2018) 提出了一个设想：如果我们采用 Transformer 编码器 (Transformer Encoder)，利用互联网上的海量句子对其进行训练，并强制它根据双向上下文来预测缺失的词语，会怎样？随后，你只需在下游任务上微调一个输出头 (Head) 即可。这种参数效率 (Parameter Efficiency) 堪称革命性的突破。

结果证明：在短短 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）便横扫了当时所有 NLP 排行榜 (Leaderboard)。到 2020 年，全球每一台搜索引擎、内容审核流水线 (Content Moderation Pipeline) 和语义搜索系统 (Semantic Search System) 内部，都运行着 BERT 模型。

到了 2026 年，纯编码器模型 (Encoder-only Models) 依然是分类、检索和结构化信息提取任务的最佳选择——其单词元 (Token) 推理速度比解码器 (Decoder) 快 5 到 10 倍，且其生成的嵌入向量构成了所有现代检索技术栈 (Retrieval Stack) 的基石。ModernBERT（2024 年 12 月发布）通过结合 Flash Attention、旋转位置编码 (Rotary Position Embedding, RoPE) 和门控线性单元 (Gated Linear Unit, GeGLU)，将该架构的上下文窗口扩展至 8K。

## 核心概念

![掩码语言建模（Masked Language Modeling）：选择词元（token），进行掩码（mask），预测原始内容](../assets/bert-mlm.svg)

### 训练信号

以句子为例：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的词元：

input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog

训练模型以预测被掩码位置上的原始词元。由于编码器（encoder）是双向的，预测位置 1 的 `[MASK]` 可以利用位置 2 及之后的 `brown fox jumps`。这正是 GPT 无法做到的。

### BERT 的掩码规则

在被选中用于预测的 15% 词元中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为随机词元。
- 10% 保持原样不变。

为什么不全部使用 `[MASK]`？因为 `[MASK]` 在推理（inference）阶段永远不会出现。如果让模型在 100% 的掩码位置上都预期看到 `[MASK]`，会在预训练（pretraining）和微调（fine-tuning）之间造成分布偏移（distribution shift）。保留 10% 的随机替换和 10% 的不变，能让模型保持“诚实”（即更贴近真实数据分布）。

### 下一句预测（Next Sentence Prediction, NSP）—— 以及为何被弃用

原始 BERT 也使用了 NSP 进行训练：给定两个句子 A 和 B，预测 B 是否紧跟在 A 之后。RoBERTa（2019）通过消融实验（ablation study）证明 NSP 反而有害无益。现代编码器已不再使用它。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文使用 2026 年的基础组件（primitives）重构了该模块：

| 组件 | 原始 BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| 位置编码（Positional） | 学习到的绝对位置编码 | RoPE |
| 激活函数（Activation） | GELU | GeGLU |
| 归一化（Normalization） | LayerNorm | Pre-norm RMSNorm |
| 注意力机制（Attention） | 全密集（Full dense） | 交替局部（128）+ 全局 |
| 上下文长度（Context length） | 512 | 8192 |
| 分词器（Tokenizer） | WordPiece | BPE |

此外，与 2018 年的技术栈不同，它原生支持 Flash-Attention。在 8K 序列长度下，其推理速度比 DeBERTa-v3 快 2–3 倍，且 GLUE 基准得分更高。

### 2026 年仍选择编码器的应用场景

| 任务 | 为何编码器优于解码器 |
|------|---------------------------|
| 检索 / 语义搜索嵌入（Embeddings） | 双向上下文 = 每个词元的嵌入质量更高 |
| 分类（情感、意图、毒性检测） | 仅需一次前向传播（forward pass）；无生成开销 |
| 命名实体识别（NER）/ 词元标注 | 逐位置输出，原生支持双向 |
| 零样本蕴含推理（NLI） | 在编码器顶部添加分类头 |
| RAG 重排序器（Reranker） | 交叉编码器（Cross-encoder）打分，速度比大语言模型重排序器快 10 倍 |

## 动手构建

### 步骤 1：掩码逻辑 (Masking Logic)

参见 `code/main.py`。函数 `create_mlm_batch` 接收一个词元 ID (Token ID) 列表、词表大小 (Vocabulary Size) 和掩码概率 (Mask Probability)。返回输入 ID (Input ID)（已应用掩码）和标签 (Label)（仅在掩码位置有值，其余位置为 -100——遵循 PyTorch 的忽略索引 (Ignore Index) 约定）。

def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels

### 步骤 2：在小型语料库上运行掩码语言模型 (Masked Language Modeling, MLM) 预测

在包含 20 个词、200 个句子的词表上训练一个 2 层编码器 (Encoder) + MLM 头 (MLM Head)。不计算梯度 (Gradient)——我们仅进行前向传播 (Forward Pass) 的健全性检查 (Sanity Check)。完整训练需要 PyTorch。

### 步骤 3：比较掩码类型

展示三路规则 (Three-way Rule) 如何使模型在未使用 `[MASK]` 的情况下仍保持可用。分别对未掩码句子和已掩码句子进行预测。两者都应生成合理的词元分布 (Token Distribution)，因为模型在训练期间已见过这两种模式。

### 步骤 4：微调头部

在玩具情感数据集 (Toy Sentiment Dataset) 上，将 MLM 头替换为分类头 (Classification Head)。仅训练头部；编码器保持冻结 (Frozen)。这是每个 BERT 应用都遵循的模式。

## 使用方法

from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)

**嵌入模型 (Embedding Model) 是微调后的 BERT。** 像 `all-MiniLM-L6-v2` 这样的 `sentence-transformers` 模型是使用对比损失 (Contrastive Loss) 训练的 BERT。编码器是相同的，只是损失函数发生了变化。

**交叉编码器重排序器 (Cross-encoder Reranker) 也是微调后的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 格式上进行配对分类 (Pair-classification)。查询 (Query) 与文档 (Document) 之间的双向注意力 (Bidirectional Attention) 正是交叉编码器在质量上优于双编码器 (Biencoder) 的关键所在。

**2026 年何时不应选择 BERT。** 任何生成式 (Generative) 任务。编码器没有合理的方式自回归地 (Autoregressively) 生成词元。此外：参数量低于 10 亿 (1B) 的任务，小型解码器 (Decoder) 能以更高的灵活性达到同等质量（如 Phi-3-Mini、Qwen2-1.5B）。

## 部署上线

参见 `outputs/skill-bert-finetuner.md`。该技能 (Skill) 为新的分类或信息抽取 (Extraction) 任务界定了 BERT 微调 (Fine-tune) 的范围（包括骨干网络 (Backbone) 选择、头部规格 (Head Spec)、数据、评估 (Evaluation) 和早停策略 (Stopping)）。

## 练习

1. **简单。** 运行 `code/main.py` 并打印 10,000 个词元（token）上的掩码（mask）分布。确认约有 15% 的词元被选中，且其中约 80% 会被替换为 `[MASK]`。
2. **中等。** 实现全词掩码（whole-word masking）：若一个词被分词为多个子词（subword），则要么将所有子词一并掩码，要么全部保留。在包含 500 个句子的语料库上评估该方法是否能提升掩码语言模型（Masked Language Modeling, MLM）的准确率。
3. **困难。** 使用公开数据集中的 10,000 个句子训练一个微型（2 层，隐藏层维度 d=64）的 BERT 模型。针对 SST-2 情感分类任务，对 `[CLS]` 词元进行微调（fine-tuning）。在参数量匹配的仅解码器（decoder-only）基线模型上进行对比——哪个表现更优？

## 关键术语

| 术语 | 通俗理解 | 实际含义 |
|------|----------|----------|
| MLM (掩码语言模型) | “掩码语言建模” | 训练信号：随机将 15% 的词元替换为 `[MASK]`，并预测原始词元。 |
| Bidirectional (双向) | “双向关注” | 编码器（encoder）的注意力机制无因果掩码（causal mask）——每个位置均可感知所有其他位置。 |
| `[CLS]` | “池化词元” | 添加至每个序列起始处的特殊词元；其最终嵌入（embedding）用作句子级表征。 |
| `[SEP]` | “序列分隔符” | 用于分隔成对序列（如查询/文档、句子 A/B）。 |
| NSP (下一句预测) | “下一句预测” | BERT 的第二个预训练任务；在 RoBERTa 中被证实无效，2019 年后已被弃用。 |
| Fine-tuning (微调) | “适配下游任务” | 保持编码器主体参数冻结；仅在顶部训练一个小型任务头（head）以处理下游任务（downstream task）。 |
| Cross-encoder (交叉编码器) | “重排序器” | 一种同时接收查询与文档作为输入的 BERT 模型，用于输出相关性得分。 |
| ModernBERT | “2024 年重构版” | 采用旋转位置编码（RoPE）、均方根层归一化（RMSNorm）、门控线性单元（GeGLU）、交替局部/全局注意力机制及 8K 上下文窗口（context）重新构建的编码器。 |

## 延伸阅读

- [Devlin 等人 (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu 等人 (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；弃用了 NSP。
- [Clark 等人 (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 在同等计算量下，替换词元检测（replaced-token detection）优于 MLM。
- [Warner 等人 (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 标准编码器参考实现。