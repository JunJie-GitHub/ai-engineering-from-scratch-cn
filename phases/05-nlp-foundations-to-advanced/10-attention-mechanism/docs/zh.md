# 注意力机制（Attention Mechanism）—— 突破性进展

> 解码器不再费力地紧盯压缩后的摘要，而是开始直接查看完整的源序列。此后的发展，无非是注意力机制与工程实现的叠加。

**类型：** 构建
**语言：** Python
**前置知识：** 第 5 阶段 · 09（序列到序列模型（Sequence-to-Sequence Models））
**预计时间：** 约 45 分钟

## 问题所在

第 09 课以一种明确的失败告终。在一个简易复制任务上训练的 GRU 编码器-解码器（GRU Encoder-Decoder），在序列长度为 5 时准确率为 89%，但当长度增加到 80 时，准确率骤降至接近随机猜测的水平。其根本原因在于结构缺陷，而非训练过程中的 bug：编码器提取的所有信息都必须塞进一个固定大小的隐藏状态（Hidden State）中，而解码器再也无法看到其他任何信息。

2014 年，Bahdanau、Cho 和 Bengio 发表了一个仅用三行代码就能解决的方案。与其只将编码器的最终状态提供给解码器，不如保留编码器的每一个状态。在解码器的每一步，计算编码器状态的加权平均值，其中的权重回答了这样一个问题：“解码器此刻需要多大程度地关注编码器位置 `i`？”这个加权平均值就是上下文向量（Context Vector），并且它会在解码器的每一步动态变化。

这就是核心思想。Transformer 架构对其进行了扩展。自注意力机制（Self-Attention）将其应用于单一序列。多头注意力机制（Multi-Head Attention）则实现了并行计算。但 2014 年的原始版本已经打破了信息瓶颈，一旦掌握了它，向 Transformer 的演进就只是工程实现问题，而非概念上的跨越。

## 核心概念

![Bahdanau 注意力机制：解码器查询所有编码器状态](../assets/attention.svg)

在解码器的每一步 `t`：

1. 将上一步的解码器隐藏状态 `s_{t-1}` 作为**查询（Query）**。
2. 将其与每一个编码器隐藏状态 `h_1, ..., h_T` 进行打分。每个编码器位置对应一个标量分数。
3. 对分数进行 Softmax 操作，得到总和为 1 的注意力权重 `α_{t,1}, ..., α_{t,T}`。
4. 计算上下文向量 `c_t = Σ α_{t,i} * h_i`。即编码器状态的加权平均值。
5. 解码器接收 `c_t` 与上一个输出的词元（Token），生成下一个词元。

加权平均正是该机制的精髓。当解码器需要将 "Je" 翻译为 "I" 时，它会给 "Je" 对应的编码器状态分配高权重，其余状态则为低权重。当需要处理 "not" 时，则会给 "pas" 分配高权重。上下文向量在每一步都会动态调整解码器的关注焦点。

## 张量形状（Shapes）（最容易让人踩坑的地方）

这是所有注意力机制（Attention Mechanism）实现首次尝试时最容易出错的地方。请仔细阅读。

| 名称 | 形状 | 备注 |
|-------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 若为双向 LSTM（BiLSTM），则 `d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 单个向量 |
| 注意力分数 `e_{t,i}` | 标量（scalar） | 每个编码器位置对应一个 |
| 注意力权重 `α_{t,i}` | 标量（scalar） | 对所有 `i` 进行 softmax 后得到 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态形状相同 |

**Bahdanau（加性，additive）分数。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的形状为 `(d_s,)`，`h_i` 的形状为 `(d_h,)`。
- `W_a` 的形状为 `(d_attn, d_s)`。`U_a` 的形状为 `(d_attn, d_h)`。
- 它们在 `tanh` 内部的求和结果形状为 `(d_attn,)`。
- `v_α` 的形状为 `(d_attn,)`。与 `v_α` 进行内积运算后，结果会坍缩为一个标量。**这正是 `v_α` 的作用。** 它并非什么魔法，而是一个投影操作，负责将注意力维度的向量转换为标量分数。

**Luong（乘性，multiplicative）分数。** 包含三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。这是一个硬性约束。如果你的编码器是双向的，请跳过此选项。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 的形状为 `(d_s, d_h)`。解除了维度必须相等的约束。
- `concat`：本质上与 Bahdanau 形式相同。由于前两种计算开销更低，因此很少使用。

**一个值得注意的 Bahdanau / Luong 易错点。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的解码器状态）。Luong 使用 `s_t`（生成*之后*的状态）。将两者混淆会导致梯度出现细微错误，且极难调试。请选定一篇论文并严格遵循其约定。

## 动手实现

### 步骤 1：加性注意力（Additive Attention）

import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

请对照上表核对各变量的形状。`encoder_states` 的形状为 `(T_enc, d_h)`。`projected_enc` 的形状为 `(T_enc, d_attn)`。`projected_dec` 的形状为 `(d_attn,)`，并会触发广播机制（Broadcast）。`combined` 的形状为 `(T_enc, d_attn)`。`scores` 的形状为 `(T_enc,)`。`weights` 的形状为 `(T_enc,)`。`context` 的形状为 `(d_h,)`。代码已就绪，可以交付了。

### 步骤 2：Luong 点积与通用注意力（Dot & General Attention）

def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights

每个函数仅需三行代码。这正是 Luong 的论文广受认可的原因。在大多数任务上精度持平，但代码量大幅精简。

### 步骤 3：数值计算示例

假设给定三个编码器状态（大致对应 "cat"、"sat"、"mat"），以及一个与第一个状态对齐程度最高的解码器状态，此时注意力分布会集中在位置 0。如果解码器状态发生偏移，转而与最后一个状态对齐，注意力则会转移到位置 2。上下文向量（Context Vector）会随之动态跟踪。

H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))

weights: [0.464 0.305 0.231]

第一行的权重最高。接着，将解码器状态向第三个编码器状态靠近，观察权重的变化。就是这么简单。注意力机制本质上就是显式对齐（Explicit Alignment）。

### 步骤 4：为何这是通往 Transformer 的桥梁

将上述概念映射到 Q/K/V 术语中：

- **查询（Query）** = 解码器状态 `s_{t-1}`
- **键（Key）** = 编码器状态（用于计算匹配得分）
- **值（Value）** = 编码器状态（用于加权求和）

在经典注意力机制中，键（Key）和值（Value）是相同的。自注意力（Self-Attention）将它们分离开来：你可以让序列与自身进行查询，并为 K 和 V 使用不同的可学习投影（Learned Projections）。多头注意力（Multi-Head Attention）则通过不同的可学习投影并行执行该过程。Transformer 架构将这一完整阶段多次堆叠，并彻底抛弃了循环神经网络（RNN）。

数学原理相同，张量形状相同。从 Bahdanau 注意力到缩放点积注意力（Scaled Dot-Product Attention）的概念过渡，主要仅在于符号表示的差异。

## 动手实践

PyTorch 和 TensorFlow 直接内置了注意力机制（Attention Mechanism）。

import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)

torch.Size([2, 5, 128]) torch.Size([2, 5, 10])

这是一个 Transformer 注意力层（Transformer Attention Layer）。查询（Query）批次包含 5 个位置，键/值（Key/Value）批次包含 10 个位置，每个向量维度为 128，共 8 个注意力头（Attention Heads）。`output` 是经过上下文增强的新查询向量。`weights` 是一个 5x10 的对齐矩阵（Alignment Matrix），可供可视化。

### 经典注意力机制（Classical Attention）依然适用的场景

- 教学用途。基于循环神经网络（RNN）的单头单层版本能让每个概念都清晰可见。
- Transformer 架构不适用的设备端序列任务。
- 2014 至 2017 年间的文献。如果不了解 Bahdanau 的约定，你将无法正确解读这些论文。
- 机器翻译（MT）中的细粒度对齐分析。原始注意力权重（Raw Attention Weights）即使在 Transformer 模型上也是一种可解释性工具，而解读它们的前提是理解其本质。

### “将注意力权重视为解释”的陷阱

注意力权重看起来具备可解释性。它们在各个位置上的权重之和为 1；你可以将其绘制成图表；数值越高意味着“模型关注了此处”。审稿人通常很喜欢这类结果。

但它们并不像看起来那样具备可解释性。Jain 和 Wallace（2019）的研究表明，在某些任务中，注意力分布可以被置换或替换为任意其他分布，而不会改变模型的预测结果。除非经过消融实验（Ablation Study）或反事实检验（Counterfactual Check），否则切勿将注意力权重作为模型推理依据进行报告。

## 交付上线

保存为 `outputs/prompt-attention-shapes.md`：

---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.

## 练习

1. **简单。** 实现 `softmax` 掩码（masking），使编码器（encoder）中的填充词元（padding tokens）获得零注意力权重（attention weight）。在包含变长序列（variable-length sequences）的批次（batch）上进行测试。
2. **中等。** 在 Luong `general` 形式中加入多头注意力（multi-head attention）。将 `d_h` 划分为 `n_heads` 个组，对每个注意力头（attention head）独立计算注意力，随后进行拼接（concatenation）。验证单头（single-head）情况下的结果是否与你先前的实现相匹配。
3. **困难。** 在第 09 课的简易复制任务（toy copy task）上，训练一个结合 Bahdanau 注意力（Bahdanau attention）的 GRU 编码器-解码器（GRU encoder-decoder）。绘制准确率（accuracy）随序列长度（sequence length）变化的曲线。将其与无注意力基线（no-attention baseline）进行对比。你应该会观察到随着序列长度增加，两者差距逐渐拉大，从而证实注意力机制有效突破了性能瓶颈（bottleneck）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 注意力（Attention） | 关注事物 | 值序列（value sequence）的加权平均，权重由查询（query）与键（key）的相似度计算得出。 |
| 查询、键、值（Query, Key, Value） | QKV | 三种线性投影：Q 负责提问，K 负责匹配，V 负责返回内容。 |
| 加性注意力（Additive attention） | Bahdanau | 前馈网络打分公式：`v^T tanh(W q + U k)`。 |
| 乘性注意力（Multiplicative attention） | Luong 点积 / 通用形式 | 打分公式为 `q^T k` 或 `q^T W k`。计算成本更低，且在大多数任务上准确率相当。 |
| 对齐矩阵（Alignment matrix） | 漂亮的可视化图 | 以 `(T_dec, T_enc)` 网格形式呈现的注意力权重。通过观察它可以了解模型关注了哪些部分。 |

## 扩展阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 原始论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) —— 三种打分变体及其对比。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) —— 关于可解释性的注意事项。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) —— 基于 PyTorch 的可运行逐步教程。