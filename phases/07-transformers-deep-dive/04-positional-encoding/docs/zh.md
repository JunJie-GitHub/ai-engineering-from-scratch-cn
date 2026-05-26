# 位置编码（Positional Encoding）—— 正弦位置编码（Sinusoidal）、旋转位置编码（RoPE）、注意力线性偏置（ALiBi）

> 注意力机制（Attention）具有排列不变性（Permutation-Invariant）。若缺乏位置信号，“The cat sat on the mat”与“mat the on sat cat the”将产生完全相同的输出。三种算法解决了这一问题——它们各自对“位置”的定义做出了不同的假设。

**类型：** 构建
**语言：** Python
**前置知识：** 第 7 阶段 · 02（自注意力机制（Self-Attention）），第 7 阶段 · 03（多头注意力机制（Multi-Head Attention））
**耗时：** 约 45 分钟

## 问题所在

缩放点积注意力（Scaled Dot-Product Attention）对顺序不敏感。注意力矩阵 `softmax(Q K^T / √d) V` 仅基于成对相似度进行计算。打乱输入 `X` 的行，输出矩阵的行也会以相同方式被打乱。注意力机制内部没有任何组件会关注位置信息。

在词袋模型（Bag-of-Words Model）中，这不算缺陷。但对于语言、代码、音频、视频——任何顺序承载语义的模态而言，这却是致命的。

解决之道在于以某种方式将位置信息注入嵌入向量（Embeddings）中。对此，业界经历了三个阶段的演进：

1. **绝对正弦编码（Absolute Sinusoidal）**（Vaswani, 2017）。将位置的正弦/余弦值直接叠加到嵌入向量中。实现简单、无需训练参数，但在超出训练长度时外推能力较差。
2. **RoPE —— 旋转位置编码（Rotary Position Embeddings）**（Su, 2021）。根据位置按比例旋转查询向量（Q）和键向量（K）。在点积运算中直接编码*相对*位置。在 2026 年占据主导地位。
3. **ALiBi —— 注意力线性偏置（Attention with Linear Biases）**（Press, 2022）。完全跳过位置嵌入；根据距离为每个注意力头（Attention Head）的注意力分数（Attention Scores）添加线性惩罚项。具备极佳的外推能力。

截至 2026 年，几乎所有前沿开源模型均采用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦编码已成为历史。

## 核心概念

![绝对正弦编码 vs RoPE 旋转 vs ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对正弦位置编码 (Absolute Sinusoidal Positional Encoding)

预先计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

在注意力机制 (Attention Mechanism) 之前执行 `X' = X + PE[:N]`。每个维度都是一个不同频率的正弦波 (Sinusoid)。模型通过学习相位模式来读取位置信息。当序列长度超过 `max_len` 时会失效：因为模型在训练时只见过 0–2047 的位置，没有任何信息告诉它在位置 2048 会发生什么。

### 旋转位置编码 (Rotary Position Embedding, RoPE)

对查询 (Query) 和键 (Key) 向量进行旋转（而非对嵌入向量 Embedding）。对于维度对 `(2i, 2i+1)`：

[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default

对位置为 `pos_k` 的键向量应用相同的旋转。点积 (Dot Product) `q'_m · k'_n` 将仅变为 `(m - n)` 的函数。也就是说：**注意力得分 (Attention Score) 仅取决于相对距离**，尽管旋转操作是基于绝对位置进行的。这是一个非常巧妙的设计。

扩展 RoPE：可以通过缩放 `base` 参数（如 NTK-aware、YaRN、LongRoPE 等方法）在不重新训练的情况下外推 (Extrapolate) 到更长的上下文 (Context)。Llama 3 正是通过这种方式将上下文长度从 8K 扩展到了 128K。

### 注意力线性偏置 (Attention with Linear Biases, ALiBi)

跳过嵌入技巧，直接对注意力得分施加偏置：

attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|

其中 `m_h` 是特定于注意力头 (Attention Head) 的斜率（例如 `1 / 2^(8·h/H)`）。距离较近的词元 (Token) 会得到增强，距离较远的词元会受到惩罚。训练阶段无额外开销。原论文表明，其长度外推能力优于正弦编码，并在原始训练长度上与 RoPE 表现相当。

### 2026 年如何选择

| 变体 | 外推能力 | 训练成本 | 采用模型 |
|---------|---------------|---------------|---------|
| 绝对正弦编码 | 较差 | 无 | 原始 Transformer、早期 BERT |
| 学习型绝对编码 | 无 | 极低 | GPT-2、GPT-3 |
| RoPE | 配合缩放后表现良好 | 无 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 极佳 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 极佳 | 无 | BLOOM、MPT、Baichuan |

RoPE 胜出的原因在于：它无需修改架构即可无缝集成到注意力机制中，能够编码相对位置，并且其 `base` 超参数 (Hyperparameter) 为长上下文微调 (Fine-tuning) 提供了一个清晰可控的调节旋钮。

## 动手实践

### 步骤 1：正弦位置编码 (Sinusoidal Encoding)

参见 `code/main.py`。仅需 4 行计算：

def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe

在第一个注意力层 (Attention Layer) 之前，将此结果加到嵌入矩阵 (Embedding Matrix) 中。

### 步骤 2：将旋转位置编码 (RoPE) 应用于 Q 和 K

RoPE 会对 Q 和 K 进行原地 (In-place) 操作。针对每一对维度 (Dimensions)：

def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out

关键点：对位置 `m` 的 Q 和位置 `n` 的 K 应用相同的函数。它们的点积 (Dot Product) 会在每个坐标对上引入一个 `cos((m-n)·θ_i)` 因子。注意力机制 (Attention) 因此能够以零额外代价学习到相对位置 (Relative Position) 信息。

### 步骤 3：ALiBi 的斜率与偏置 (Bias)

def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax

将 `bias[h]` 加到第 `h` 个注意力头的 `(seq_len, seq_len)` 注意力分数矩阵 (Attention Score Matrix) 中，然后进行 Softmax 操作。

### 步骤 4：验证 RoPE 的相对距离特性

随机选取两个向量 `a, b`。先按 `(pos_a, pos_b)` 进行旋转，再按 `(pos_a + k, pos_b + k)` 进行旋转。两次计算得到的点积必须在浮点误差 (Floating-point Error) 范围内保持一致。这一特性正是 RoPE 的核心所在——它对绝对偏移 (Absolute Offset) 具有平移不变性，仅相对间距 (Relative Gap) 起作用。

## 实际应用

PyTorch 2.5+ 已在 `torch.nn.functional` 中内置了 RoPE 工具函数。大多数生产环境代码会使用 `flash_attn` 或 `xformers`，其中 RoPE 直接在注意力内核 (Attention Kernel) 内部完成计算。

from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}

**2026 年长上下文 (Long-Context) 技巧：**

- **NTK 感知插值 (NTK-aware Interpolation)。** 当上下文长度从 4K 扩展至 16K+ 时，将 `base` 重新缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 一种更智能的插值方法，能够在长上下文中保持注意力熵 (Attention Entropy)。Llama 3.1 128K 模型采用了该技术。
- **LongRoPE。** 微软于 2024 年提出的方法，利用进化搜索 (Evolutionary Search) 为每个维度选取独立的缩放因子。Phi-3-Long 模型使用了该技术。
- **位置插值 (Position Interpolation) + 微调 (Fine-tuning)。** 只需按扩展因子压缩位置索引，并使用 10 亿至 50 亿 (1–5B) 个词元 (Tokens) 进行微调。效果出奇地好。

## 部署上线

请参阅 `outputs/skill-positional-encoding-picker.md`。该技能会根据目标上下文长度（context length）、外推需求（extrapolation needs）以及训练预算（training budget），为新模型挑选合适的编码策略（encoding strategy）。

## 练习

1. **简单。** 将正弦位置编码（Positional Encoding, `PE`）矩阵绘制为热力图，参数设为 `max_len=512, d=128`。验证“随着维度索引增大，条纹逐渐变宽”的规律。
2. **中等。** 实现 NTK感知（NTK-aware）的旋转位置嵌入（Rotary Position Embedding, RoPE）缩放。在长度为 256 的序列上训练一个小型语言模型（Language Model, LM），然后在长度为 1024 的序列上进行测试（分别使用和不使用缩放）。测量困惑度（perplexity）。
3. **困难。** 在同一个注意力模块（attention module）中同时实现注意力线性偏置（Attention with Linear Biases, ALiBi）和 RoPE。在长度为 512 的序列上，使用复制任务（copy task）训练一个 4 层 Transformer。在测试时将序列外推（extrapolate）至 2048。对比两者的性能下降（degradation）情况。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 位置编码（Positional Encoding） | “告诉注意力机制顺序信息” | 添加到词嵌入（embeddings）或注意力机制中，用于编码位置信息的任意信号。 |
| 正弦编码（Sinusoidal） | “最初的那个” | 将几何频率的 `sin/cos` 函数值加到词嵌入中；不具备外推能力。 |
| RoPE | “旋转嵌入” | 根据位置相关的角度旋转查询向量（Q）和键向量（K）；点积结果编码相对距离。 |
| ALiBi | “线性偏置技巧” | 在注意力分数上加上 `-m·|i-j|`；无需额外嵌入，外推性能极佳。 |
| base | “RoPE 的调节旋钮” | RoPE 中的频率缩放因子；在推理时增大该值可扩展上下文长度。 |
| NTK-aware | “一种 RoPE 缩放技巧” | 重新缩放 `base` 参数，使得上下文扩展时高频维度不会被过度压缩。 |
| YaRN | “高级玩法” | 逐维度进行插值与外推，以保持注意力熵（attention entropy）稳定。 |
| 外推（Extrapolation） | “在训练长度之外依然有效” | 位置编码方案能否在超出训练时见过的 `max_len` 后，依然输出正确结果？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦位置编码。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 原始论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi 论文。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 目前最先进的 RoPE 缩放方法。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 关于 Llama 2 长上下文的论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 微软提出的方法，已被 Phi-3-Long 采用，并在“使用指南（Use It）”部分引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 所有 RoPE 缩放方案（默认、线性、动态、YaRN、LongRoPE、Llama-3）的生产级实现代码。