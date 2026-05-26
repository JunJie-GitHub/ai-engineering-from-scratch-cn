# 序列到序列模型 (Sequence-to-Sequence Models)

> 两个伪装成翻译器的循环神经网络 (RNN)。它们遭遇的瓶颈正是注意力机制 (Attention) 诞生的原因。

**类型：** 构建实践
**语言：** Python
**前置知识：** 第 5 阶段 · 08（用于文本的卷积神经网络与循环神经网络），第 3 阶段 · 11（PyTorch 入门）
**预计耗时：** 约 75 分钟

## 问题背景

分类任务将可变长度序列映射为单一标签。翻译任务则将一个可变长度序列映射为另一个可变长度序列。输入与输出位于不同的词表 (vocabulary) 中，甚至可能是不同的语言，且两者长度之间没有任何必然的对应关系。

序列到序列架构 (seq2seq architecture)（Sutskever, Vinyals, Le, 2014）通过一种刻意简化的方案攻克了这一难题。它仅使用两个循环神经网络 (RNN)。其中一个负责读取源句子并生成一个固定大小的上下文向量 (context vector)；另一个则读取该向量，并逐个词元 (token) 生成目标句子。这与你第 08 课编写的代码完全相同，只是拼接方式不同。

研究该架构有两个重要原因。首先，上下文向量瓶颈是自然语言处理 (NLP) 领域最具教学价值的“失败案例”。正是它催生了注意力机制 (Attention) 和 Transformer 架构所擅长的所有技术。其次，其训练方案（教师强制 (Teacher Forcing)、计划采样 (Scheduled Sampling)、推理时的束搜索 (Beam Search)）至今仍适用于包括大语言模型 (LLM) 在内的所有现代生成系统。

## 核心概念

**编码器 (Encoder)。** 一个用于读取源句子的 RNN。其最终的隐藏状态即为**上下文向量 (context vector)**——对整个输入信息的固定大小摘要。理论上，它除了源信息外不会丢失任何内容。

**解码器 (Decoder)。** 另一个由上下文向量初始化的 RNN。在每一步中，它将上一步生成的词元作为输入，并输出目标词表上的概率分布。通过采样或 `argmax` 来选择下一个词元，并将其反馈回网络作为下一步的输入。重复此过程，直到生成 `<EOS>` 词元或达到最大长度限制。

**训练 (Training)：** 在解码器的每一步计算交叉熵损失 (Cross-Entropy Loss)，并在整个序列上求和。通过两个网络进行标准的随时间反向传播 (Backpropagation Through Time)。

**教师强制 (Teacher Forcing)。** 在训练期间，解码器在步骤 `t` 的输入是位置 `t-1` 的*真实标签 (Ground-Truth)* 词元，而非解码器自身上一步的预测结果。这能稳定训练过程；若不采用此方法，早期的错误会级联放大，导致模型无法收敛。而在推理阶段，你必须使用模型自身的预测结果，因此训练与推理之间始终存在分布差异。这种差异被称为**暴露偏差 (Exposure Bias)**。

**瓶颈 (The Bottleneck)。** 编码器从源句子中学到的所有信息都必须被压缩进那一个上下文向量中。长句子会丢失细节，生僻词的特征会被模糊化。语序调整（例如法语的 *chat noir* 与英语的 *black cat*）只能依靠死记硬背，而非通过计算推导。

注意力机制 (Attention)（第 10 课）通过允许解码器查看编码器的*每一个*隐藏状态（而不仅仅是最后一个）来解决此问题。这就是它的核心思想。

## 动手实践

### 步骤 1：编码器 (Encoder)

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden

`outputs` 的形状为 `[batch, seq_len, hidden_dim]`，表示每个输入位置对应一个隐藏状态 (hidden state)。`hidden` 的形状为 `[1, batch, hidden_dim]`，代表最后一步的状态。第 08 课曾提到“对输出进行池化以进行分类”。在这里，我们保留最后一个隐藏状态作为上下文向量 (context vector)，并忽略每一步的输出。

### 步骤 2：解码器 (Decoder)

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden

解码器每次调用仅处理一步。输入：一批单个词元 (token) 和当前的隐藏状态。输出：下一个词元的词汇表 logits 以及更新后的隐藏状态。

### 步骤 3：带教师强制 (Teacher Forcing) 的训练循环

def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len

这里有两个值得注意的可调节参数。`ignore_index=0` 用于在计算损失时跳过填充词元 (padding token)。`teacher_forcing_ratio` 表示在每一步使用真实词元而非模型预测词元的概率。训练初期可设为 1.0（完全教师强制），随后在训练过程中逐渐衰减至约 0.5，以缩小暴露偏差 (exposure bias) 带来的差距。

### 步骤 4：推理循环（贪婪解码）

@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)

贪婪解码 (Greedy decoding) 在每一步都选择概率最高的词元。它容易产生累积误差：一旦选定某个词元，便无法撤回。**束搜索 (Beam search)** 会保留得分最高的 `k` 个部分序列，并在最后选出得分最高的完整序列。通常束宽 (beam width) 设为 3 到 5。

### 步骤 5：瓶颈演示

在一个简单的复制任务上训练模型：源序列为 `[a, b, c, d, e]`，目标序列为 `[a, b, c, d, e]`。逐步增加序列长度，观察准确率的变化。

seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%

单个 GRU 隐藏状态无法无损地记忆长达 40 个词元的输入。虽然编码器每一步都包含相关信息，但解码器只能看到最后一个状态。注意力机制 (Attention) 可以直接解决这一问题。

## 使用指南

PyTorch 提供了基于 `nn.Transformer` 和 `nn.LSTM` 的序列到序列（seq2seq）模板。Hugging Face 的 `transformers` 库则内置了完整的编码器-解码器（encoder-decoder）模型（如 BART、T5、mBART、NLLB），这些模型均在数十亿词元（token）上进行了训练。

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))

现代的编码器-解码器架构已摒弃循环神经网络（RNN），转而采用 Transformer。其高层架构（编码器、解码器、逐词元生成）与 2014 年的 seq2seq 论文完全一致，但每个模块内部的机制已有所不同。

### 何时仍应考虑使用基于 RNN 的 seq2seq

对于新项目而言，几乎永远不需要。特定例外情况包括：

- 流式翻译场景：需要以有限内存逐词元处理输入。
- 端侧文本生成场景：Transformer 的内存开销过高。
- 教学目的。理解编码器-解码器的瓶颈是弄懂 Transformer 为何胜出的最快途径。

### 暴露偏差（Exposure Bias）及其缓解方法

- **计划采样（Scheduled Sampling）**。在训练期间逐渐降低教师强制（Teacher Forcing）比例，使模型学会从自身错误中恢复。
- **最小风险训练（Minimum Risk Training）**。基于句子级 BLEU 分数而非词元级交叉熵（Cross-Entropy）进行训练。这更贴近你的实际优化目标。
- **强化学习微调（Reinforcement Learning Fine-tuning）**。使用特定指标奖励序列生成器。现代大语言模型（LLM）的基于人类反馈的强化学习（RLHF）即采用此方法。

上述三种方法同样适用于基于 Transformer 的生成任务。

## 部署上线

保存为 `outputs/prompt-seq2seq-design.md`：

---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).

## 练习

1. **简单。** 实现一个示例级复制任务。在目标序列与源序列相同的输入-输出对上训练门控循环单元（GRU）序列到序列（Seq2Seq）模型。分别测量序列长度为 5、10 和 20 时的准确率，并复现该性能瓶颈。
2. **中等。** 添加束宽（beam width）为 3 的束搜索（Beam Search）解码。在小型平行语料库上，与贪婪解码（Greedy Decoding）进行对比并测量 BLEU 分数。记录束搜索表现更优的场景（通常出现在序列末尾的词元/Token）以及两者效果无差异的场景。
3. **困难。** 在包含 1 万对样本的释义数据集上微调 `facebook/bart-base` 模型。在预留（Held-out）测试集上，对比微调后模型采用束宽为 4 的解码输出与原始基础模型的输出。报告 BLEU 分数，并精选 10 个用于定性分析的示例。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器（Encoder） | 输入端 RNN | 读取源序列。生成每一步的隐藏状态以及最终的上下文向量。 |
| 解码器（Decoder） | 输出端 RNN | 由上下文向量初始化。逐个生成目标词元（Token）。 |
| 上下文向量（Context Vector） | 摘要信息 | 编码器最终的隐藏状态。尺寸固定。这正是注意力机制（Attention）所要解决的瓶颈。 |
| 教师强制（Teacher Forcing） | 使用真实词元 | 在训练时输入真实的前一个词元（Ground-truth）。有助于稳定学习过程。 |
| 暴露偏差（Exposure Bias） | 训练/测试差异 | 基于真实词元训练的模型从未练习过如何从自身生成的错误中恢复。 |
| 束搜索（Beam Search） | 更优的解码方式 | 在每一步保留 top-k 个部分序列，而非贪婪地只选择单一最优路径。 |

## 扩展阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始的 Seq2Seq 论文。仅四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了 GRU 以及编码器-解码器（Encoder-Decoder）架构范式。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 提出注意力机制的论文。建议在本课程结束后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可实际运行的 Seq2Seq + 注意力机制代码教程。