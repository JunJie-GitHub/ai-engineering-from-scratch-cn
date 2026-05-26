# 面向文本的卷积神经网络与循环神经网络 (CNNs and RNNs for Text)

> 卷积学习 n元语法 (n-grams)。循环结构负责记忆。两者均已被注意力机制 (Attention) 取代，但在算力受限的硬件上依然具有重要价值。

**类型：** 构建实践
**语言：** Python
**前置知识：** 第3阶段 · 11（PyTorch 简介）、第5阶段 · 03（词嵌入）、第4阶段 · 02（从零实现卷积）
**预计耗时：** 约 75 分钟

## 问题背景

TF-IDF 与 Word2Vec 生成的扁平向量忽略了词序。基于它们构建的分类器无法区分 `dog bites man` 与 `man bites dog`。而词序有时恰恰承载着关键信号。

在 Transformer 出现之前，两类架构填补了这一空白。

**面向文本的卷积网络 (TextCNN)。** 在词嵌入序列上应用一维卷积 (1D Convolutions)。宽度为 3 的卷积核 (Filter) 相当于一个可学习的三元语法 (Trigram) 检测器：它覆盖三个词并输出一个得分。堆叠不同宽度（2、3、4、5）的卷积核以检测多尺度模式。通过最大池化 (Max-pooling) 得到固定大小的表示。结构扁平、并行计算、速度极快。

**循环神经网络 (RNN、LSTM、GRU)。** 逐个处理词元 (Tokens)，并维护一个向前传递信息的隐藏状态 (Hidden State)。具有序列性、记忆能力，且支持灵活的输入长度。在 2014 至 2017 年间主导了序列建模 (Sequence Modeling) 领域，随后注意力机制崛起。

本节将动手实现这两种架构，并指出促使注意力机制诞生的核心缺陷。

## 核心原理

**TextCNN**（Kim, 2014）。词元首先被转换为嵌入向量。宽度为 `k` 的一维卷积将卷积核在连续的 `k`-gram 嵌入序列上滑动，生成特征图 (Feature Map)。对该特征图进行全局最大池化，提取出最强的激活值。将不同宽度卷积核的最大池化输出进行拼接，随后输入分类器头部 (Classifier Head)。

为何有效。每个卷积核本质上都是一个可学习的 n-gram 检测器。最大池化具有位置不变性 (Position-invariant)，因此无论“not good”出现在评论的开头还是中间，都会触发相同的特征。三种宽度各配置 100 个卷积核，即可得到 300 个学习到的 n-gram 检测器。训练过程完全并行，无序列依赖。

**RNN。** 在每个时间步 `t`，隐藏状态的计算公式为 `h_t = f(W * x_t + U * h_{t-1} + b)`。权重 `W`、`U` 和偏置 `b` 在所有时间步间共享。时间步 `T` 的隐藏状态是对整个前缀序列的摘要。用于分类时，可对 `h_1 ... h_T` 进行池化操作（取最大值、均值或最后一个状态）。

基础 RNN 容易遭遇梯度消失 (Vanishing Gradients) 问题。**LSTM** 引入了门控机制，用于决定遗忘什么、存储什么以及输出什么，从而在长序列中稳定梯度。**GRU** 将 LSTM 简化为两个门控；在参数量更少的情况下表现相近。

**双向循环神经网络 (Bidirectional RNNs)** 同时运行一个正向和一个反向 RNN，并将两者的隐藏状态拼接。每个词元的表示都能同时捕获左右两侧的上下文。这对序列标注任务 (Tagging Tasks) 至关重要。

## 动手实现

### 步骤 1：在 PyTorch 中构建 TextCNN（文本卷积神经网络）

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))

`transpose(1, 2)` 操作将张量形状从 `[batch, seq_len, embed_dim]` 转换为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道（channel）。无论输入序列长度如何，池化（pooling）后的输出均为固定尺寸。

### 步骤 2：LSTM（长短期记忆网络）分类器

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))

此处对序列执行最大池化（max-pooling），而非仅取最后一个时间步的状态。在分类任务中，最大池化通常优于直接提取最后一个隐藏状态（hidden state），因为长序列末尾的信息往往会主导最终状态，从而掩盖前面的关键特征。

### 步骤 3：梯度消失（vanishing gradient）演示（直观理解）

未引入门控机制（gating mechanism）的普通循环神经网络（RNN）难以学习长距离依赖（long-range dependencies）。考虑一个示例任务：判断标记 `A` 是否出现在序列的任意位置。若 `A` 位于第 1 位，且序列总长为 100 个标记，则损失（loss）计算出的梯度需反向传播，并经历 99 次循环权重（recurrent weight）的连续相乘。若该权重小于 1，梯度将逐渐消失（vanish）；若大于 1，梯度则会爆炸（explode）。

def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.

LSTM 通过引入**细胞状态（cell state）**解决了该问题。该状态在网络中传递时主要依赖加法交互（尽管遗忘门（forget gate）会对其进行乘法缩放，但梯度仍能沿这条“高速公路”顺畅流动）。GRU（门控循环单元）以更少的参数实现了类似机制。两者均能支持 100 步以上长序列的稳定训练。

### 步骤 4：为何这仍然不够

即便使用了 LSTM，仍有三个问题未能解决。

1. **序列瓶颈（Sequential bottleneck）。** 在长度为 1000 的序列上训练 RNN 需要执行 1000 次串行的前向/反向传播步骤。无法在时间维度上进行并行化（parallelize）。
2. **编码器-解码器（encoder-decoder）架构中的固定尺寸上下文向量（context vector）。** 解码器只能看到编码器最终的隐藏状态，该状态是对整个输入序列的压缩表示。对于长输入，细节信息会大量丢失。第 09 课将直接探讨此问题。
3. **长距离依赖准确率瓶颈（Distant-dependency accuracy ceiling）。** LSTM 的性能虽优于普通 RNN，但在跨越 200 步以上的序列传播特定信息时依然力不从心。

注意力机制（Attention）一举解决了上述三个问题。Transformer 架构则彻底摒弃了循环结构（recurrence）。第 10 课将是整个课程的关键转折点。

## 使用场景

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 已具备生产就绪（production-ready）条件。训练代码遵循标准范式。

Hugging Face 提供了预训练词嵌入（pretrained embeddings），你可以直接将其作为输入层接入：

from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))

根据约束条件选择架构的检查清单：

- **边缘/端侧推理（Edge / on-device inference）。** 结合 GloVe 词嵌入的 TextCNN 模型体积比 Transformer 小 10 到 100 倍。如果你的部署目标是手机，这就是首选技术栈。
- **流式/在线分类（Streaming / online classification）。** RNN 每次处理一个词元（token），而 Transformer 需要完整的序列。对于实时输入的文本，LSTM 依然具有优势。
- **用于基线（baselines）的轻量级模型。** 在新任务上快速迭代。在 CPU 上只需 5 分钟即可训练一个 TextCNN。
- **数据有限时的序列标注（Sequence labeling）。** BiLSTM-CRF（第 06 课）在 1k-10k 条标注句子的规模下，依然是生产级的命名实体识别（NER）架构。

其余场景均推荐使用 Transformer。

## 交付部署

保存为 `outputs/prompt-text-encoder-picker.md`：

---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.

## 练习

1. **简单。** 在一个三分类的玩具数据集（需自行构造数据）上训练文本卷积神经网络（TextCNN）。验证使用多种卷积核宽度（2, 3, 4）在平均 F1 分数上是否优于单一宽度（3）。
2. **中等。** 为长短期记忆网络（LSTM）分类器实现最大池化（max-pool）、平均池化（mean-pool）和末状态池化（last-state pooling）。在小型数据集上进行对比；记录哪种池化策略表现最佳，并推测其原因。
3. **困难。** 构建一个双向长短期记忆网络-条件随机场（BiLSTM-CRF）命名实体识别（NER）标注器（结合第 06 课与本课内容）。在 CoNLL-2003 数据集上进行训练。将其与第 06 课中仅使用条件随机场（CRF）的基线模型以及经过微调的 BERT 模型进行对比。报告训练时间、内存占用和 F1 分数。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 文本卷积神经网络（TextCNN） | 用于文本的 CNN | 在词嵌入上堆叠一维卷积，并配合全局最大池化。源自 Kim (2014)。 |
| 循环神经网络（RNN） | 循环网络 | 隐藏状态在每个时间步更新：`h_t = f(W x_t + U h_{t-1})`。 |
| 长短期记忆网络（LSTM） | 门控 RNN | 增加了输入门、遗忘门、输出门以及细胞状态。能够在长序列中稳定训练。 |
| 门控循环单元（GRU） | 简化版 LSTM | 仅包含两个门而非三个。精度相近，但参数量更少。 |
| 双向（Bidirectional） | 双向处理 | 将前向与后向 RNN 的输出拼接。每个词元都能同时看到其上下文的两侧信息。 |
| 梯度消失（Vanishing gradient） | 训练信号衰减 | 在普通 RNN 中，权重反复乘以小于 1 的数值，导致早期时间步的梯度趋近于零。 |

## 扩展阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 的原始论文。全文仅八页，通俗易懂。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 的奠基论文。行文出乎意料地清晰透彻。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 凭借直观的图解，让 LSTM 变得对所有人而言都易于理解。