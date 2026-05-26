# 从零实现自注意力机制 (Self-Attention)

> 注意力机制 (Attention) 就像一个查找表，其中的每个词元 (Token) 都在询问“谁与我相关？”，并自行学习得出答案。

**类型：** 构建实践
**编程语言：** Python
**前置知识：** 第 3 阶段（深度学习核心），第 5 阶段第 10 课（序列到序列模型 (Sequence-to-Sequence)）
**预计耗时：** 约 90 分钟

## 学习目标

- 仅使用 NumPy 从零实现缩放点积自注意力机制 (Scaled Dot-Product Self-Attention)，包括查询/键/值 (Query/Key/Value) 投影以及基于 Softmax 的加权求和
- 构建多头注意力层 (Multi-Head Attention Layer)，实现注意力头拆分、并行注意力计算以及结果拼接
- 追踪注意力矩阵 (Attention Matrix) 如何捕获词元之间的关系，并解释为何除以 `sqrt(d_k)` 进行缩放能防止 Softmax 饱和
- 应用因果掩码 (Causal Masking) 将双向注意力转换为自回归（解码器风格）注意力

## 问题背景

循环神经网络 (RNN) 每次只能逐个处理序列中的词元。当处理到第 50 个词元时，第 1 个词元的信息已经历了 50 次压缩步骤。长距离依赖关系被挤压进固定大小的隐藏状态中——这是一个无论使用多少长短期记忆网络 (LSTM) 门控机制都无法彻底解决的瓶颈。

2014 年 Bahdanau 等人提出的注意力机制论文给出了解决方案：让解码器回看编码器的每一个位置，并自行判断哪些位置对当前步骤至关重要。但该机制仍然依附于 RNN 之上。2017 年的《Attention Is All You Need》论文提出了一个更尖锐的问题：如果注意力机制是*唯一*的机制会怎样？无需循环结构，无需卷积，仅靠注意力。

自注意力机制允许序列中的每个位置在单次并行计算中关注所有其他位置。这正是 Transformer 架构 (Transformer) 具备高速、可扩展性并占据主导地位的原因。

## 核心概念

### 数据库查找类比

可以将注意力 (Attention) 想象成一种“软性”的数据库查找：

Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values

每个词元 (Token) 都会生成三个向量：
- **查询向量 (Query, Q)**：“我在寻找什么？”
- **键向量 (Key, K)**：“我包含什么内容？”
- **值向量 (Value, V)**：“如果被选中，我能提供什么信息？”

查询向量与所有键向量进行点积 (Dot Product) 运算，得到注意力分数 (Attention Scores)。分数越高，表示“该键与我的查询越匹配”。这些分数随后用于对值向量进行加权。最终的输出即为值向量的加权和。

### Q、K、V 的计算

每个词元嵌入 (Token Embedding) 都会通过三个可学习的权重矩阵进行投影：

Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value

以单个词元为例，其可视化表示如下：

             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"

### 注意力矩阵

当计算出所有词元的 Q、K、V 后，注意力分数将构成一个矩阵：

Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence

### 为什么要进行缩放？

点积的大小会随着维度 dk 的增加而增长。如果 dk = 64，点积值可能达到几十的量级，从而将 Softmax 推入梯度消失 (Gradient Vanishing) 的区域。解决方法是：除以 sqrt(dk)。

Scaled scores = (Q @ K^T) / sqrt(dk)

这样可以确保数值处于 Softmax 能够产生有效梯度的范围内。

### Softmax 将分数转化为权重

Softmax 函数将原始分数转换为每一行的概率分布：

Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)

至此，每个词元都获得了一组权重，用于表示它应该对其他每个词元分配多少注意力。

### 值向量的加权和

每个词元的最终输出是所有值向量的加权和：

output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5

### 完整流程

                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+

单行公式表示如下：

Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V


## 动手实现

### 步骤 1：从零实现 Softmax（Softmax）

Softmax 函数（Softmax）将原始 logits（Logits）转换为概率分布。为保持数值稳定性（Numerical Stability），计算时需先减去最大值。

import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")

### 步骤 2：缩放点积注意力（Scaled Dot-Product Attention）

这是核心函数。接收查询（Query）、键（Key）、值（Value）矩阵，并返回注意力输出（Attention Output）与权重矩阵（Weight Matrix）。

def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights

### 步骤 3：带可学习投影的自注意力类（Self-Attention）

一个完整的自注意力（Self-Attention）模块，其 Wq、Wk、Wv 权重矩阵采用类 Xavier 缩放（Xavier-like Scaling）策略进行初始化。

class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights

### 步骤 4：在句子序列上运行

为句子生成模拟的词嵌入（Embeddings），并观察注意力权重（Attention Weights）的分布。

sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()

### 步骤 5：使用 ASCII 热力图可视化注意力

将注意力权重映射为字符，以便快速直观地查看结果。

def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)

## 实际应用

PyTorch 的 `nn.MultiheadAttention` 实现了与我们构建的完全相同的功能，并额外增加了多头拆分（multi-head splitting）和输出投影（output projection）：

import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))

核心区别在于：多头注意力（multi-head attention）会并行运行多个注意力函数，每个头都拥有独立的 Q、K、V 投影矩阵，其维度为 `dk = d_model / n_heads`，随后将各头结果进行拼接。这使得模型能够同时捕捉不同类型的依赖关系。

## 交付成果

本课程的产出包括：
- `outputs/prompt-attention-explainer.md` - 用于通过数据库查找类比来解释注意力机制的提示词（prompt）

## 练习

1. 修改 `scaled_dot_product_attention` 函数，使其接受一个可选的掩码矩阵（mask matrix），在 softmax 操作前将特定位置的值设为负无穷（这正是因果掩码/解码器掩码（causal/decoder masking）的工作原理）
2. 从零实现多头注意力：将 Q、K、V 拆分为 `n_heads` 个块，分别对每个块执行注意力计算，拼接结果后，通过最终的权重矩阵 Wo 进行投影
3. 选取两个长度相同但内容不同的句子，将它们输入同一个 `SelfAttention` 实例，并对比它们的注意力模式。哪些部分发生了变化？哪些保持不变？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Query (Q) | “查询向量” | 输入经过可学习投影得到的向量，代表该词元（token）正在检索何种信息 |
| Key (K) | “标签向量” | 经过可学习投影得到的向量，代表该词元包含的信息，用于与 Query 进行匹配 |
| Value (V) | “内容向量” | 经过可学习投影得到的向量，携带实际信息，将根据注意力得分进行加权聚合 |
| Scaled dot-product attention（缩放点积注意力） | “注意力公式” | `softmax(QK^T / sqrt(dk)) @ V` —— 缩放操作可防止高维空间中 softmax 函数进入饱和区（softmax saturation） |
| Self-attention（自注意力） | “词元同时关注自身与其他词元” | Q、K、V 均源自同一序列的注意力机制，允许序列中任意位置关注其他所有位置 |
| Attention weights（注意力权重） | “关注程度” | 对缩放点积结果应用 softmax 后生成的位置概率分布 |
| Multi-head attention（多头注意力） | “并行注意力” | 使用不同投影矩阵并行执行多个注意力计算，随后拼接结果以获取更丰富的特征表示 |

## 延伸阅读

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - Transformer (Transformer) 架构的原始论文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - 完整架构的最佳可视化图解
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - 附带逐行注释与详解的 PyTorch 实现