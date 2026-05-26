# 预训练小型 GPT（1.24 亿参数）

> GPT-2 Small 拥有 1.24 亿个参数。它包含 12 个 Transformer 层（Transformer Layers）、12 个注意力头（Attention Heads）以及 768 维的嵌入向量（Embeddings）。你可以在单张 GPU 上从零开始训练它，仅需数小时。大多数人从未这样做过，他们直接使用预训练检查点（Pre-trained Checkpoints）。但如果你不亲自训练一次，就无法真正理解你所依赖构建产品的模型内部究竟在发生什么。

**类型：** 构建
**语言：** Python（使用 numpy）
**前置要求：** 第 10 阶段，课程 01-03（分词器（Tokenizers）、构建分词器、数据流水线（Data Pipelines））
**耗时：** 约 120 分钟

## 学习目标

- 从零实现完整的 GPT-2 架构（1.24 亿参数）：词元嵌入（Token Embeddings）、位置嵌入（Positional Embeddings）、Transformer 块（Transformer Blocks）以及语言模型头（Language Model Head）
- 使用交叉熵损失（Cross-Entropy Loss）和下一词元预测（Next-Token Prediction）在文本语料库上训练 GPT 模型
- 实现自回归文本生成（Autoregressive Text Generation），包含温度采样（Temperature Sampling）与 top-k/top-p 过滤（Top-k/Top-p Filtering）
- 监控训练损失曲线（Training Loss Curves），并验证模型是否学习到了连贯的语言模式

## 问题所在

你知道 Transformer 是什么。你看过那些架构图。你能背诵 "attention is all you need"，还能在白板上画出标有“多头注意力（Multi-Head Attention）”的方框。

但这并不意味着你真正理解模型在生成文本时内部究竟发生了什么。

GPT-2 Small（采用权重绑定（Weight Tying））共有 124,438,272 个参数。其中每一个参数都是通过运行训练循环（Training Loop）确定的：前向传播（Forward Pass）、计算损失（Compute Loss）、反向传播（Backward Pass）、更新权重（Update Weights）。12 个 Transformer 块，每个块包含 12 个注意力头，768 维的嵌入空间，以及包含 50,257 个词元的词表（Vocabulary）。每当模型生成一个词元时，这 1.24 亿个参数都会参与到一个单一的矩阵乘法链（Matrix Multiplication Chain）中，将词元 ID 序列转化为下一个词元的概率分布。

如果你从未亲手构建过它，你实际上是在操作一个黑盒（Black Box）。你可以调用 API，可以进行微调（Fine-tuning）。但当出现问题时——比如模型产生幻觉（Hallucination）、不断重复内容，或拒绝遵循指令——你将完全无法从心智模型（Mental Model）上理解*为什么*会这样。

本课程将带你从零构建 GPT-2 Small。不使用 PyTorch，而是使用 numpy。每一次矩阵乘法都清晰可见，每一个梯度都由你的代码亲自计算。你将亲眼见证这 1.24 亿个数字是如何协同工作以预测下一个词的。

## 核心概念

### GPT 架构

GPT 是一种自回归语言模型（Autoregressive Language Model）。“自回归”意味着它每次生成一个词元（Token），且每个词元的生成都以之前所有的词元为条件。该架构由多个 Transformer 解码器块（Transformer Decoder Blocks）堆叠而成。

以下是从词元 ID 到下一个词元概率的完整计算图：

1. 输入词元 ID。形状（Shape）：(batch_size, seq_len)。
2. 词元嵌入查找（Token Embedding Lookup）。每个 ID 映射为一个 768 维向量。形状：(batch_size, seq_len, 768)。
3. 位置嵌入查找（Position Embedding Lookup）。每个位置（0, 1, 2, ...）映射为一个 768 维向量。形状相同。
4. 将词元嵌入与位置嵌入相加。
5. 通过 12 个 Transformer 块。
6. 最终层归一化（Layer Normalization）。
7. 线性投影至词表大小。形状：(batch_size, seq_len, vocab_size)。
8. 应用 Softmax 函数获取概率。

这就是整个模型的全部。没有卷积（Convolutions），没有循环结构（Recurrence）。仅仅是嵌入层、注意力机制、前馈网络（Feedforward Networks）和层归一化堆叠了 12 次。

graph TD
    A["Token IDs\n(batch, seq_len)"] --> B["Token Embeddings\n(batch, seq_len, 768)"]
    A --> C["Position Embeddings\n(batch, seq_len, 768)"]
    B --> D["Add"]
    C --> D
    D --> E["Transformer Block 1"]
    E --> F["Transformer Block 2"]
    F --> G["..."]
    G --> H["Transformer Block 12"]
    H --> I["Layer Norm"]
    I --> J["Linear Head\n(768 -> 50257)"]
    J --> K["Softmax\nNext-token probabilities"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#0f3460,color:#fff
    style C fill:#1a1a2e,stroke:#0f3460,color:#fff
    style D fill:#1a1a2e,stroke:#16213e,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
    style I fill:#1a1a2e,stroke:#16213e,color:#fff
    style J fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#51cf66,color:#fff

### Transformer 块

这 12 个块中的每一个都遵循相同的模式。采用前置归一化架构（Pre-norm Architecture）（GPT-2 使用前置归一化，而非原始 Transformer 中的后置归一化 Post-norm）：

1. 层归一化（LayerNorm）
2. 多头自注意力机制（Multi-Head Self-Attention）
3. 残差连接（Residual Connection）（将输入加回）
4. 层归一化（LayerNorm）
5. 前馈网络（Feed-Forward Network / MLP）
6. 残差连接（将输入加回）

残差连接至关重要。如果没有它们，在反向传播（Backpropagation）过程中，梯度在到达第 1 个块时就会消失。有了它们，梯度可以通过“跳跃”路径直接从损失函数流向任意层。这就是为什么你可以堆叠 12、32 甚至 96 个块（传闻 GPT-4 使用了 120 个块）。

### 注意力机制：核心机制

自注意力机制（Self-Attention）允许每个词元查看所有先前的词元，并决定对每个词元的关注程度。以下是其数学原理。

对于每个词元位置，从输入中计算三个向量：
- **查询向量（Query, Q）**：“我在寻找什么？”
- **键向量（Key, K）**：“我包含什么信息？”
- **值向量（Value, V）**：“我携带什么信息？”

Q = input @ W_q    (768 -> 768)
K = input @ W_k    (768 -> 768)
V = input @ W_v    (768 -> 768)

attention_scores = Q @ K^T / sqrt(d_k)
attention_scores = mask(attention_scores)   # causal mask: -inf for future positions
attention_weights = softmax(attention_scores)
output = attention_weights @ V

因果掩码（Causal Mask）正是使 GPT 具备自回归特性的关键。位置 5 可以关注位置 0-5，但不能关注位置 6、7、8 等。这防止了模型在训练期间通过“偷看”未来词元来作弊。

**多头注意力机制（Multi-Head Attention）**将 768 维空间拆分为 12 个头，每个头为 64 维。每个头学习不同的注意力模式。一个头可能追踪句法关系（如主谓一致），另一个可能追踪语义相似性（如同义词），还有一个可能追踪位置邻近性（如相邻词汇）。所有 12 个头的输出会被拼接并投影回 768 维。

graph LR
    subgraph MultiHead["Multi-Head Attention (12 heads)"]
        direction TB
        I["Input (768)"] --> S1["Split into 12 heads"]
        S1 --> H1["Head 1\n(64 dims)"]
        S1 --> H2["Head 2\n(64 dims)"]
        S1 --> H3["..."]
        S1 --> H12["Head 12\n(64 dims)"]
        H1 --> C["Concat (768)"]
        H2 --> C
        H3 --> C
        H12 --> C
        C --> O["Output Projection\n(768 -> 768)"]
    end

    subgraph SingleHead["Each Head Computes"]
        direction TB
        Q["Q = X @ W_q"] --> A["scores = Q @ K^T / 8"]
        K["K = X @ W_k"] --> A
        A --> M["Apply causal mask"]
        M --> SM["Softmax"]
        SM --> MUL["weights @ V"]
        V["V = X @ W_v"] --> MUL
    end

    style I fill:#1a1a2e,stroke:#e94560,color:#fff
    style O fill:#1a1a2e,stroke:#e94560,color:#fff
    style Q fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#0f3460,color:#fff
    style V fill:#1a1a2e,stroke:#0f3460,color:#fff

除以 sqrt(d_k)（即 sqrt(64) = 8）是一种缩放操作（Scaling）。如果没有它，高维向量的点积会变得非常大，从而将 Softmax 推入梯度几乎为零的区域。这是原始论文《Attention Is All You Need》中的关键洞见之一。

### KV 缓存：为何推理速度如此之快

在训练期间，你会一次性处理整个序列。而在推理（Inference）期间，你每次生成一个词元。如果不进行优化，生成第 N 个词元需要重新计算所有前 N-1 个词元的注意力。这意味着每生成一个词元的复杂度为 O(N^2)，对于长度为 N 的序列，总复杂度为 O(N^3)。

KV 缓存（KV Cache）解决了这个问题。在计算完每个词元的 K 和 V 后，将它们存储起来。当生成第 N+1 个词元时，你只需计算新词元的 Q，并查找所有先前词元缓存的 K 和 V。这将 K 和 V 计算的单词元成本从 O(N) 降低到了 O(1)。注意力分数的计算仍然是 O(N)，因为你仍需关注所有先前的位置，但你避免了对输入进行冗余的矩阵乘法运算。

对于具有 12 层和 12 个头的 GPT-2，KV 缓存为每个词元存储 2 (K + V) x 12 层 x 12 头 x 64 维 = 18,432 个值。对于 1024 个词元的序列，在 FP32 精度下约为 75MB。对于具有 128 层的 Llama 3 405B，单个序列的 KV 缓存可能超过 10GB。这就是为什么长上下文推理是内存受限（Memory-Bound）的。

### 预填充与解码：推理的两个阶段

当你向大语言模型（LLM）发送提示词（Prompt）时，推理过程分为两个截然不同的阶段。

**预填充（Prefill）**阶段并行处理你的整个提示词。由于所有词元都是已知的，模型可以同时计算所有位置的注意力。此阶段是计算受限（Compute-Bound）的——GPU 正以全吞吐量进行矩阵乘法运算。在 A100 上处理 1000 个词元的提示词，预填充大约需要 20-50 毫秒。

**解码（Decode）**阶段逐个生成词元。每个新词元都依赖于所有先前的词元。此阶段是内存受限的——瓶颈在于从 GPU 内存中读取模型权重和 KV 缓存，而非矩阵运算本身。GPU 的计算核心大部分时间处于空闲状态，等待内存读取。对于 GPT-2，无论矩阵乘法需要多少 FLOPs，每个解码步骤所需的时间大致相同，因为内存带宽是主要限制因素。

这种区分对生产系统至关重要。预填充的吞吐量随 GPU 计算能力扩展（更多的 FLOPS = 更快的预填充）。解码的吞吐量随内存带宽扩展（更快的内存 = 更快的解码）。这就是为什么 NVIDIA 的 H100 相较于 A100 更注重内存带宽的提升——它能直接加速词元生成。

graph LR
    subgraph Prefill["Phase 1: Prefill"]
        direction TB
        P1["Full prompt\n(all tokens known)"]
        P2["Parallel computation\n(compute-bound)"]
        P3["Builds KV Cache"]
        P1 --> P2 --> P3
    end

    subgraph Decode["Phase 2: Decode"]
        direction TB
        D1["Generate token N"]
        D2["Read KV Cache\n(memory-bound)"]
        D3["Append to KV Cache"]
        D4["Generate token N+1"]
        D1 --> D2 --> D3 --> D4
        D4 -.->|repeat| D1
    end

    Prefill --> Decode

    style P1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P3 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D4 fill:#1a1a2e,stroke:#e94560,color:#fff

### 训练循环

训练大语言模型本质上是下一个词元预测（Next-Token Prediction）。给定词元序列 [0, 1, 2, ..., N-1]，预测词元序列 [1, 2, 3, ..., N]。损失函数（Loss Function）是模型预测的概率分布与实际下一个词元之间的交叉熵（Cross-Entropy）。

一个训练步骤包含：

1. **前向传播（Forward Pass）**：将批次数据通过所有 12 个块。获取每个位置的 logits（Softmax 前的分数）。
2. **计算损失**：计算 logits 与目标词元（输入序列向后偏移一个位置）之间的交叉熵。
3. **反向传播（Backward Pass）**：使用反向传播算法计算所有 1.24 亿参数的梯度。
4. **优化器步骤（Optimizer Step）**：更新权重。GPT-2 使用 Adam 优化器，并配合学习率预热（Learning Rate Warmup）和余弦衰减（Cosine Decay）。

学习率调度（Learning Rate Schedule）的重要性可能超出你的预期。GPT-2 在前 2000 个步骤中将学习率从 0 预热至峰值，随后按余弦曲线衰减。一开始使用高学习率会导致模型发散（Diverge）。在训练后期保持恒定的高学习率则会引起震荡。这种“先预热后衰减”的模式已被所有主流大语言模型采用。

### GPT-2 Small：具体参数

| 组件 | 形状 | 参数量 |
|-----------|-------|------------|
| 词元嵌入 | (50257, 768) | 38,597,376 |
| 位置嵌入 | (1024, 768) | 786,432 |
| 每块注意力权重 (W_q, W_k, W_v, W_out) | 4 x (768, 768) | 2,359,296 |
| 每块前馈网络 (up + down) | (768, 3072) + (3072, 768) | 4,718,592 |
| 每块层归一化 (2x) | 2 x 768 x 2 | 3,072 |
| 最终层归一化 | 768 x 2 | 1,536 |
| **每块总计** | | **7,080,960** |
| **总计（12 个块）** | | **85,054,464 + 39,383,808 = 124,438,272** |

输出投影层（Logits Head）与词元嵌入矩阵共享权重。这被称为权重绑定（Weight Tying）——它将参数量减少了 3800 万，并提升了模型性能，因为它强制模型在输入和输出中使用相同的表示空间（Representation Space）。

## 构建

### 步骤 1：嵌入层 (Embedding Layer)

词元嵌入 (Token Embedding) 将 50,257 个可能的词元中的每一个映射为一个 768 维向量。位置嵌入 (Position Embedding) 则补充了每个词元在序列中的位置信息。两者相加得到最终的输入表示。

import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb

初始化时采用 0.02 的标准差源自 GPT-2 论文。若标准差过大，初始的前向传播 (Forward Pass) 会产生极端值，导致训练不稳定；若过小，所有输入的初始输出将几乎完全相同，使得早期的梯度信号失去作用。

### 步骤 2：带因果掩码的自注意力机制 (Self-Attention with Causal Mask)

首先实现单头注意力 (Single-Head Attention)。因果掩码 (Causal Mask) 会在 Softmax 操作前将未来位置的值设为负无穷，从而确保每个位置只能关注自身及其之前的位置。

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, -1, -2 if Q.ndim == 4 else 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V

此处的 Softmax 实现在指数运算前减去了最大值。若不这样做，`exp(大数)` 会溢出为无穷大。这是一种保证数值稳定性 (Numerical Stability) 的技巧，它不会改变输出结果，因为对于任意常数 c，都有 `softmax(x - c) = softmax(x)`。

### 步骤 3：多头注意力机制 (Multi-Head Attention)

将 768 维的输入拆分为 12 个注意力头 (Attention Head)，每个头 64 维。每个头独立计算注意力，随后将结果拼接 (Concatenate) 并投影回 768 维。

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_out = np.random.randn(embed_dim, embed_dim) * 0.02

    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        Q = (x @ self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return attn_out @ self.W_out

“重塑-转置-重塑” (Reshape-Transpose-Reshape) 的操作是多头的注意力机制中最令人困惑的部分。具体过程如下：形状为 `(batch, seq_len, 768)` 的张量先变为 `(batch, seq_len, 12, 64)`，再转置为 `(batch, 12, seq_len, 64)`。此时，12 个头各自拥有一个 `(seq_len, 64)` 的矩阵用于计算注意力。注意力计算完成后，我们逆向执行该过程：`(batch, 12, seq_len, 64)` 变回 `(batch, seq_len, 12, 64)`，最终重塑为 `(batch, seq_len, 768)`。

### 步骤 4：Transformer 块 (Transformer Block)

一个完整的 Transformer 块包含：层归一化 (LayerNorm)、带残差连接 (Residual Connection) 的多头注意力、层归一化、带残差连接的前馈网络 (Feedforward Network)。

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)  # GELU approximation: ReLU for simplicity
        return h @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x

前馈网络将 768 维的输入扩展至 3,072 维（4 倍），应用非线性激活函数后，再投影回 768 维。这种“扩展-收缩”模式为模型在每个位置提供了更“宽”的内部表示空间。GPT-2 使用的是 GELU 激活函数，但为了简化代码，此处使用 ReLU——在理解整体架构时，两者的差异微乎其微。

### 步骤 5：完整 GPT 模型 (Full GPT Model)

堆叠 12 个 Transformer 块。在模型前端添加嵌入层，在后端添加输出投影层。

class MiniGPT:
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        logits = x @ self.embedding.token_embed.T
        return logits

    def count_parameters(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for block in self.blocks:
            total += block.attn.W_q.size + block.attn.W_k.size
            total += block.attn.W_v.size + block.attn.W_out.size
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total

注意这里的权重共享 (Weight Tying)：`logits = x @ self.embedding.token_embed.T`。输出投影复用了词元嵌入矩阵（的转置）。这不仅仅是一种节省参数的技巧，它意味着模型在理解词元（嵌入）和预测词元（输出）时使用的是同一个向量空间。

### 步骤 6：训练循环 (Training Loop)

若要真正训练一个 1.24 亿参数的模型，你需要 GPU 和 PyTorch。此处的训练循环旨在通过纯 NumPy 运行的小型模型来演示其底层机制。为了便于计算，我们使用了一个微型模型（4 层、4 个头、128 维）。

def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
    return loss


def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=200, lr=3e-4):
    tokens = np.array(list(text.encode("utf-8")[:2048]))
    model = MiniGPT(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len, ff_dim=embed_dim * 4
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training tokens: {len(tokens):,}")
    print(f"Config: {num_layers} layers, {num_heads} heads, {embed_dim} dims")
    print()

    for step in range(num_steps):
        start_idx = np.random.randint(0, max(1, len(tokens) - seq_len - 1))
        batch_tokens = tokens[start_idx:start_idx + seq_len + 1]

        input_ids = batch_tokens[:-1].reshape(1, -1)
        target_ids = batch_tokens[1:].reshape(1, -1)

        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return model

初始损失值接近 `ln(vocab_size)`——对于包含 256 个词元的字节级词表而言，即 `ln(256) = 5.55`。随机初始化的模型会为每个词元分配相等的概率。随着训练的进行，损失值会逐渐下降，因为模型学会了预测常见模式：例如 "t" 后面常跟 "h"，句号后面常跟空格等。

在实际生产环境中，你会使用 Adam 优化器 (Adam Optimizer)，并结合梯度累积 (Gradient Accumulation)、学习率预热 (Learning Rate Warmup) 和梯度裁剪 (Gradient Clipping)。前向传播-计算损失-反向传播-参数更新的循环逻辑完全一致，只是优化器的实现更为复杂。

### 步骤 7：文本生成 (Text Generation)

文本生成利用训练好的模型逐个预测词元。每次预测都从输出分布中采样，或直接贪婪地选择概率最高的词元（Argmax）。

def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / temperature
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

    return tokens

温度参数 (Temperature) 用于控制随机性。温度为 1.0 时使用原始分布；温度为 0.5 会使分布更尖锐（更具确定性——模型更倾向于选择高概率词元）；温度为 1.5 会使分布更平坦（更随机——低概率词元获得更大的被选中机会）；温度为 0.0 则对应贪婪解码 (Greedy Decoding)，即始终选择概率最高的词元。

使用 `tokens[-seq_len:]` 窗口是必要的，因为模型存在最大上下文长度限制（GPT-2 为 1024）。一旦超出该限制，就必须丢弃最旧的词元。这就是大家常说的“上下文窗口 (Context Window)”。

## 使用它

### 完整训练与生成演示

corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""

model = train_mini_gpt(corpus, num_steps=200)

prompt = list("The transformer".encode("utf-8"))
output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.8)
generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
print(f"\nGenerated: {generated_text}")

在小型语料库（corpus）和小型模型上，生成的文本最多只能达到半连贯的水平。它会从训练文本中学习一些字节级模式（byte-level patterns），但无法像 GPT-2 那样利用 40GB 训练数据和完整的 1.24 亿参数架构（parameter architecture）进行泛化（generalize）。重点不在于输出质量，而在于你可以追踪每一个步骤：嵌入查找（embedding lookup）、注意力计算（attention computation）、前馈变换（feedforward transformation）、logit 投影（logit projection）、Softmax 以及采样（sampling）。每一个操作都是清晰可见的。

## 交付上线

本教程将生成 `outputs/prompt-gpt-architecture-analyzer.md` 文件——这是一个用于分析任意 GPT 风格模型架构选择（architecture choices）的提示词（prompt）。只需向其输入模型卡片（model card）或技术报告（technical report），它便会详细拆解参数分配（parameter allocation）、注意力设计（attention design）以及扩展决策（scaling decisions）。

## 练习

1. 修改模型，将层数（layers）和注意力头数（attention heads）分别改为 24 和 16，而非原来的 12/12。统计参数量（parameters）。将模型深度（depth）加倍与将宽度（width，即嵌入维度 embedding dimension）加倍相比，效果有何不同？

2. 实现 GELU 激活函数（GELU activation function）(GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))))，并替换前馈网络（feedforward network）中的 ReLU。分别使用这两种激活函数进行 500 步训练，并比较最终的损失值（loss）。

3. 在生成函数中添加 KV 缓存（KV cache）。在首次前向传播（forward pass）后，存储每一层的 K 和 V 张量（tensors），并在生成后续词元（tokens）时重复使用它们。测量加速效果：分别在启用和未启用缓存的情况下生成 200 个词元，并比较实际运行时间（wall-clock time）。

4. 实现 top-k 采样（top-k sampling，仅考虑概率最高的 k 个词元）和 top-p 采样（top-p sampling，即核采样 nucleus sampling：考虑累积概率超过 p 的最小词元集合）。在温度（temperature）为 0.8 的条件下，对比 top-k=50 与 top-p=0.95 的输出质量。

5. 构建一个训练损失曲线（training loss curve）绘制工具。训练模型 1000 步，并绘制损失值随步数变化的曲线。识别其中的三个阶段：初期快速下降（学习常见字节）、中期缓慢下降（学习字节模式）以及平台期（在小语料库上出现过拟合）。无论你训练的是 128 维模型还是 GPT-4，该曲线的形状都是相同的。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 自回归（Autoregressive） | “它一次生成一个词” | 每个输出词元（token）都以所有先前的词元为条件——模型预测 P(token_n \| token_0, ..., token_{n-1}) |
| 因果掩码（Causal mask） | “它看不到未来” | 一个由负无穷大（-infinity）值组成的上三角矩阵，用于在训练期间防止模型关注未来的位置 |
| 多头注意力（Multi-head attention） | “多种注意力模式” | 将 Q、K、V 拆分为并行的多个头（例如 GPT-2 中 12 个头，每个头 64 维），使每个头能够学习不同类型的关系 |
| KV 缓存（KV Cache） | “用于加速的缓存” | 存储先前词元计算得到的 Key 和 Value 张量，以避免在自回归生成过程中进行重复计算 |
| 预填充（Prefill） | “处理提示词” | 推理的第一阶段，所有提示词（prompt）词元被并行处理——该阶段受 GPU 浮点运算能力（FLOPS）限制（计算密集型） |
| 解码（Decode） | “生成词元” | 推理的第二阶段，词元逐个生成——该阶段受 GPU 内存带宽限制（访存密集型） |
| 权重共享（Weight tying） | “共享嵌入层” | 输入词元嵌入（token embeddings）和输出投影头（output projection head）使用相同的矩阵——在 GPT-2 中可节省 3800 万参数 |
| 残差连接（Residual connection） | “跳跃连接” | 将输入直接加到子层的输出上（x + sublayer(x)）——使梯度能够在深层网络中顺畅流动 |
| 层归一化（Layer normalization） | “归一化激活值” | 沿特征维度进行归一化，使其均值为 0、方差为 1，并包含可学习的缩放（scale）和偏置（bias）参数 |
| 交叉熵损失（Cross-entropy loss） | “预测有多离谱” | 对正确下一个词元分配概率的负对数（-log），并在所有位置上取平均——这是大语言模型（LLM）训练的标准目标函数 |

## 延伸阅读

- [Radford et al., 2019 -- "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) -- 介绍了参数量涵盖 1.24 亿至 15 亿模型系列的 GPT-2 论文
- [Vaswani et al., 2017 -- "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) -- 提出缩放点积注意力（Scaled Dot-Product Attention）与多头注意力（Multi-Head Attention）机制的原始 Transformer 论文
- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) -- Meta 如何利用 1.6 万块 GPU 将 GPT 架构扩展至 4050 亿参数
- [Pope et al., 2022 -- "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) -- 正式区分预填充（Prefill）与解码（Decode）阶段，并对 KV 缓存（KV Cache）展开分析的论文