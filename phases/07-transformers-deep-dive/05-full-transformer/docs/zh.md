# 完整 Transformer —— 编码器 + 解码器

> 注意力机制 (Attention) 是核心。其余组件——残差连接 (Residuals)、归一化 (Normalization)、前馈网络 (Feed-Forward) 与交叉注意力 (Cross-Attention)——均为支撑模型深层堆叠的脚手架。

**类型：** 构建
**语言：** Python
**前置知识：** 第 7 阶段 · 02（自注意力机制 (Self-Attention)）、第 7 阶段 · 03（多头注意力机制 (Multi-Head Attention)）、第 7 阶段 · 04（位置编码 (Positional Encoding)）
**预计耗时：** 约 75 分钟

## 问题所在

单一的注意力层仅能作为特征提取器，而非完整的模型。每层仅靠一次矩阵乘法 (Matrix Multiplication) 无法提供处理自然语言所需的模型容量。模型必须具备深度——但若缺乏合理的底层架构管线，堆叠深度将导致训练崩溃。

2017 年 Vaswani 等人的论文整合了六项关键设计决策，将单一的注意力层转化为可堆叠的模块。此后的所有 Transformer 架构——仅编码器型 (Encoder-Only，如 BERT)、仅解码器型 (Decoder-Only，如 GPT) 以及编码器-解码器型 (Encoder-Decoder，如 T5)——均沿用了这一基础骨架。尽管到了 2026 年，模块内部已历经诸多优化（如均方根归一化 (RMSNorm)、SwiGLU 激活函数、前置归一化 (Pre-Norm)、旋转位置编码 (RoPE)），但其核心骨架始终如一。

本课程将聚焦于这一基础骨架。后续课程将对其进行专项扩展——第 06 课针对编码器，第 07 课针对解码器，第 08 课针对编码器-解码器架构。

## 核心概念

![编码器与解码器模块内部结构，连线图](../assets/full-transformer.svg)

### 六大核心组件

1. **嵌入与位置信号（Embedding + positional signal）**。词元（Token）→ 向量。位置信息通过旋转位置编码（Rotary Position Embedding, RoPE，现代）或正弦位置编码（sinusoidal，经典）注入。
2. **自注意力机制（Self-attention）**。每个位置都会关注所有其他位置。在解码器中会使用掩码（Masked）。
3. **前馈神经网络（Feed-forward network, FFN）**。逐位置的双层多层感知机（Multilayer Perceptron, MLP）：`W_2 · activation(W_1 · x)`。默认扩展比（Expansion ratio）为 4 倍。
4. **残差连接（Residual connection）**。`x + sublayer(x)`。若无此设计，超过约 6 层后梯度将消失。
5. **层归一化（Layer normalization）**。使用 `LayerNorm` 或均方根层归一化（Root Mean Square Layer Normalization, RMSNorm，现代）。用于稳定残差流（residual stream）。
6. **交叉注意力机制（Cross-attention，仅解码器）**。查询（Query）来自解码器，键（Key）和值（Value）来自编码器输出。

### 编码器模块（用于 BERT、T5 编码器）

x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘

编码器是双向的。不使用掩码。所有位置均可看到所有位置的信息。

### 解码器模块（用于 GPT、T5 解码器）

x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out

每个解码器模块包含三个子层。中间的交叉注意力层是信息从编码器流向解码器的唯一通道。在纯解码器架构（如 GPT）中，交叉注意力被省略，仅保留掩码自注意力与前馈神经网络。

### 前置归一化与后置归一化（Pre-norm vs post-norm）

原始论文中的对比：`x + sublayer(LN(x))`（后置）与 `LN(x + sublayer(x))`（前置）。后置归一化在 2019 年左右逐渐失宠——若不进行细致的预热（warmup），很难训练深层网络。前置归一化（`LN` 位于子层*之前*）已成为 2026 年的默认配置：Llama、Qwen、GPT-3+ 和 Mistral 均采用此设计。

### 2026 年现代化模块

Vaswani 等人 2017 年的原始论文采用了 LayerNorm + ReLU。现代模型堆栈已将两者替换。实际生产环境中的模块结构如下：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| 归一化（Normalization） | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展比 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量保持一致） |
| 位置编码 | 正弦绝对位置编码 | RoPE |
| 注意力机制 | 完整多头注意力（Full MHA） | 分组查询注意力（Grouped Query Attention, GQA）或多头潜在注意力（Multi-Head Latent Attention, MLA） |
| 偏置项（Bias terms） | 有 | 无 |

RMSNorm 去除了 LayerNorm 的均值中心化操作（减少一次减法运算），从而节省计算量，且经验表明其稳定性至少与 LayerNorm 相当。在 Llama、PaLM 和 Qwen 的论文中，SwiGLU（`Swish(W1 x) ⊙ W3 x`）在困惑度（perplexity, ppl）上始终比 ReLU/GELU 前馈网络低约 0.5 个点。

### 参数量计算

对于单个模块，设 `d_model = d`，FFN 扩展比为 `r`：

- MHA：`4 · d²`（Q、K、V、O 投影矩阵）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化层：可忽略不计

当 `d = 4096, r = 2.6, layers = 32`（大致对应 Llama 3 8B）时，总参数量为：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`（加上嵌入层和输出头）。这与官方公布的参数量一致。

## 动手构建

### 步骤 1：基础组件

使用第 03 课中的微型 `Matrix` 类（为保持独立性已复制到本文件中）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以均方根 (Root Mean Square, RMS)。无需减去均值。
- `gelu(x)` 以及 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 与 `decoder_block(x, enc_out, params)`。

完整连接逻辑请参见 `code/main.py`。

### 步骤 2：连接 2 层编码器与 2 层解码器

将它们堆叠起来。将编码器输出传入每个解码器的交叉注意力 (Cross-Attention) 层。在输出投影 (Output Projection) 前添加一个最终的层归一化 (Layer Normalization, LN)。

def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x

### 步骤 3：在示例数据上运行前向传播 (Forward Pass)

输入一个包含 6 个词元 (Token) 的源序列和一个包含 5 个词元的目标序列。验证输出形状是否为 `(5, vocab)`。无需训练——本课重点在于架构，而非损失函数。

### 步骤 4：替换为 RMSNorm + SwiGLU

将层归一化 (LayerNorm) 和 ReLU 前馈网络 (ReLU-FFN) 替换为 RMSNorm 和 SwiGLU。确认输出形状依然匹配。这就是 2026 年的现代化改造，仅需替换一个函数即可。

## 实际应用

PyTorch/TF 的参考实现为：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年的生产环境代码都会自行实现模块，原因如下：

- Flash Attention 直接在注意力机制内部调用，而非通过 `nn.MultiheadAttention`。
- 分组查询注意力 (Grouped-Query Attention, GQA) / 多头潜在注意力 (Multi-Head Latent Attention, MLA) 并未包含在标准库参考实现中。
- 旋转位置编码 (Rotary Position Embedding, RoPE)、RMSNorm、SwiGLU 并非 PyTorch 的默认配置。

Hugging Face `transformers` 库提供了清晰的参考模块，建议阅读：`modeling_llama.py` 是 2026 年标准的仅解码器 (Decoder-Only) 架构实现。代码约 500 行，非常值得通读一遍。

**编码器 vs 解码器 vs 编码器-解码器 —— 如何选择：**

| 需求 | 选择 | 示例 |
|------|------|------|
| 文本分类、嵌入表示、问答 | 仅编码器 (Encoder-Only) | BERT, DeBERTa, ModernBERT |
| 文本生成、对话、代码、推理 | 仅解码器 (Decoder-Only) | GPT, Llama, Claude, Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 (Encoder-Decoder) | T5, BART, Whisper |

仅解码器架构在语言模型领域胜出，是因为其扩展性最佳，且能同时处理理解与生成任务。当输入具有明确的“源序列”特征时（如翻译、语音识别、结构化任务），编码器-解码器架构依然是最佳选择。

## 交付发布

请参见 `outputs/skill-transformer-block-reviewer.md`。该技能模块会对照 2026 年的默认标准审查新的 Transformer 模块实现，并标记缺失的组件（前置归一化 Pre-Norm、RoPE、RMSNorm、GQA、前馈网络扩展比例 FFN Expansion Ratio）。

## 练习

1. **简单。** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 的配置下，统计你的 `encoder_block`（编码器块）中的参数量。通过实际实现该模块并调用 `sum(p.numel() for p in block.parameters())` 进行验证。
2. **中等。** 将架构从后归一化 (post-norm) 切换为前归一化 (pre-norm)。分别初始化两种配置，并在随机输入下测量经过 12 层堆叠后的激活范数 (activation norm)。后归一化的激活值应当会发散（爆炸）；而前归一化的激活值则应保持有界。
3. **困难。** 在一个简易的复制任务（将输入 `x` 反转后复制）上实现一个 4 层的编码器-解码器 (encoder-decoder)。训练 100 个步骤并记录损失值 (loss)。随后将组件替换为均方根归一化 (RMSNorm) + SwiGLU + 旋转位置编码 (RoPE) —— 观察损失值是否会下降？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 模块 (Block) | “一个 Transformer 层” | 归一化层 + 注意力机制 + 归一化层 + 前馈神经网络 (FFN) 的堆叠，并由残差连接包裹。 |
| 残差 (Residual) | “跳跃连接” | 输出为 `x + f(x)`；使梯度能够顺畅流经深层堆叠网络。 |
| 前归一化 (Pre-norm) | “先归一化，而非后归一化” | 现代架构：`x + sublayer(LN(x))`。无需繁琐的预热 (warmup) 操作即可训练更深的网络。 |
| 均方根归一化 (RMSNorm) | “去掉均值计算的层归一化 (LayerNorm)” | 仅除以均方根 (RMS)；减少一次运算，经验稳定性相同。 |
| SwiGLU | “大家纷纷转向的 FFN” | `Swish(W1 x) ⊙ W3 x → W2`。在语言模型困惑度 (LM ppl) 上优于 ReLU/GELU。 |
| 交叉注意力 (Cross-attention) | “解码器如何‘看到’编码器” | 多头注意力 (MHA)，其中查询 (Q) 来自解码器，键/值 (K/V) 来自编码器输出。 |
| FFN 扩展系数 (FFN expansion) | “中间多层感知机 (MLP) 有多宽” | 隐藏层维度与 `d_model` 的比值，通常为 4（配合层归一化 (LayerNorm)）或 2.6（配合 SwiGLU）。 |
| 无偏置 (Bias-free) | “去掉 +b 项” | 现代堆叠网络在线性层中省略偏置项；能轻微提升 ppl 表现，并减小模型体积。 |

## 延伸阅读

- [Vaswani 等人 (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 原始模块规范。
- [Xiong 等人 (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) —— 深入解析为何前归一化优于后归一化。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) —— RMSNorm 论文。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) —— SwiGLU 的原始论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) —— 2026 年标准的仅解码器 (decoder-only) 模块实现。