# 从零构建 Transformer —— 压轴实战项目

> 十三节课。一个模型。绝无捷径。

**类型：** 构建
**语言：** Python
**前置要求：** 第 7 阶段 · 01 至 13 课。请勿跳过。
**预计耗时：** 约 120 分钟

## 问题描述

你已经研读了所有相关论文，并亲手实现了注意力机制 (Attention)、多头拆分 (Multi-head Splits)、位置编码 (Positional Encodings)、编码器与解码器模块 (Encoder and Decoder Blocks)、BERT 与 GPT 的损失函数 (Losses)、混合专家模型 (Mixture of Experts, MoE) 以及键值缓存 (Key-Value Cache, KV Cache)。现在，是时候让它们在真实任务中协同运作了。

压轴项目：在字符级语言建模 (Character-level Language Modeling) 任务上，端到端 (End-to-End) 训练一个小型的仅解码器 Transformer (Decoder-only Transformer)。它能阅读莎士比亚的作品，也能生成全新的莎士比亚风格文本。它的规模足够小，在笔记本电脑上不到 10 分钟即可完成训练；同时它的实现足够严谨，只需替换为更大的数据集并延长训练时间，就能得到一个真正的语言模型 (Language Model, LM)。

这是本课程的“nanoGPT”项目。它并非原创——Karpathy 于 2023 年发布的 nanoGPT 教程是每个学生至少会亲手实现一遍的参考实现。我们借鉴了其整体架构，并围绕本课程已涵盖的知识点进行了重新设计与改造。

## 核心概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

架构标注如下：

input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07

### 我们提供的组件

- `GPTConfig` — 集中配置所有超参数 (hyperparameters) 的单一入口。
- `MultiHeadAttention` — 支持因果掩码 (causal mask) 与批处理，并提供可选的 Flash 风格计算路径（基于 PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代前馈神经网络 (Feed-Forward Network, FFN)。
- `Block` — 采用前置归一化 (pre-norm)，并将注意力机制与前馈网络包裹在残差连接 (residual connection) 中。
- `GPT` — 包含词嵌入 (embeddings)、堆叠的模块 (blocks)、语言模型头 (LM head) 以及 `generate()` 方法。
- 训练循环：集成 AdamW 优化器、余弦学习率调度 (cosine learning rate scheduler) 与梯度裁剪 (gradient clipping)。
- 基于莎士比亚文本的字符级分词器 (char-level tokenizer)。

### 我们未提供的组件

- RoPE — 旋转位置编码 (Rotary Position Embedding, RoPE) 在第 04 课中进行了概念性实现。为简化起见，此处使用可学习位置编码 (learned positional embeddings)。练习题要求你替换为 RoPE。
- 生成过程中的 KV 缓存 (Key-Value cache) — 每个生成步骤都会重新计算完整前缀的注意力。速度较慢但实现更简单。练习题要求你添加 KV 缓存。
- Flash Attention — 若输入匹配，PyTorch 2.0+ 会自动调度闪存注意力 (Flash Attention)；此处我们直接使用 `F.scaled_dot_product_attention`。
- MoE — 混合专家模型 (Mixture of Experts, MoE) 每个模块仅包含单个 FFN。你在第 11 课中已了解过 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，使用 `tinyshakespeare.txt` 数据集训练一个 4 层、4 头、`d_model=128` 的 GPT 模型，运行 2,000 步：

- 训练损失 (training loss) 在约 6 分钟内从 ~4.2（随机初始化）收敛至 ~1.5。
- 采样输出具有莎士比亚风格：会出现古英语词汇、换行格式以及诸如 "ROMEO:" 的专有名词。
- 验证损失 (validation loss，使用文本末尾预留的 10% 数据) 与训练损失紧密贴合；在此模型规模与计算预算下未出现过拟合 (overfitting)。

## 动手构建

本课程使用 PyTorch。请安装 `torch`（CPU 版本即可）。详见 `code/main.py`。该脚本负责处理以下任务：

- 若缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器（Byte-level char tokenizer）。
- 按 90/10 比例划分训练集与验证集（Train/val split）。
- 在支持的硬件上使用 bf16 自动转换（bf16 autocast）的训练循环（Training loop）。
- 训练完成后进行文本采样（Sampling）。

### 步骤 1：数据准备

text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)

共 65 个唯一字符。词表（Vocabulary）极小，仅需 4 字节即可容纳 `vocab_size`。无需字节对编码（BPE），也无需处理复杂的分词器（Tokenizer）问题。

### 步骤 2：模型

详见 `code/main.py`。该模块直接沿用第 05 课的标准架构——前置归一化（Pre-norm）、RMSNorm、SwiGLU 激活函数以及因果多头注意力机制（Causal MHA）。配置为 4/4/128 时，参数量约为 80 万（~800K）。

### 步骤 3：训练循环

获取一批长度为 256 的随机 Token 窗口。前向传播（Forward）。计算偏移一位的交叉熵损失（Shift-by-one cross-entropy）。反向传播（Backward）。执行 AdamW 优化器更新。记录日志。重复上述步骤。

for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

### 步骤 4：采样

给定提示词（Prompt），反复进行前向传播，从 Top-p 逻辑值（Logits）中采样，将结果追加到序列末尾并继续。生成 500 个 Token 后停止。

### 步骤 5：查看输出

训练 2,000 步后：

ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...

这并非真正的莎士比亚作品，但已具备莎士比亚的文风。仅凭约 80 万参数和笔记本电脑上 6 分钟的训练时间，这已是一个显著的胜利。

## 实际应用

本综合项目（Capstone）提供了一套参考架构。若要将其交付为实际可用的系统，可进行以下三项扩展：

1. **替换分词器（Tokenizer）。** 使用字节对编码（BPE）（例如 `tiktoken.get_encoding("cl100k_base")`）。词表大小将从 65 跃升至约 50,000。模型容量需相应扩大以进行补偿。
2. **在更大规模的语料库（Corpus）上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（来自 HuggingFace）。对于 1.25 亿参数的 GPT 模型，在单张 A100 上处理 100 亿（10B）Token 大约需要 24 小时。
3. **添加旋转位置编码（RoPE）+ 键值缓存（KV Cache）+ 闪存注意力（Flash Attention）。** 下方的练习将逐步指导你实现每一项。

最终你将得到一个拥有 1.25 亿参数、能够生成流畅英文的 GPT 模型。它虽非前沿大模型，但这条相同的代码路径——仅仅是规模更大——正是 Karpathy、EleutherAI 和艾伦人工智能研究所（Allen Institute）在 2026 年用于训练研究检查点（Checkpoints）的核心方案。

## 项目交付

详见 `outputs/skill-transformer-review.md`。该技能评估（Skill Review）将针对从零实现 Transformer（Transformer-from-scratch）的代码进行审查，验证其是否涵盖了前 13 课的所有正确性要求。

## 练习

1. **简单。** 运行 `code/main.py`。验证训练模型在最后一步的验证损失（validation loss）是否低于 2.0。将 `max_steps` 从 2,000 调整为 5,000 —— 验证损失（val loss）是否会持续下降？
2. **中等。** 将可学习的位置嵌入（positional embeddings）替换为旋转位置编码（RoPE）。在 `MultiHeadAttention` 内部对查询（Q）和键（K）应用旋转变换。重新训练模型，并验证验证损失至少不高于原有水平。
3. **中等。** 在采样循环中实现键值缓存（KV cache）。分别在启用和禁用缓存的情况下生成 500 个词元（token）。在笔记本电脑上，实际运行时间（wall-clock time）应缩短 5 至 20 倍。
4. **困难。** 为模型增加第二个输出头，用于预测下下个词元（next-plus-one token）（MTP —— 源自 DeepSeek-V3 的多词元预测 Multi-Token Prediction）。进行联合训练。观察其是否带来性能提升？
5. **困难。** 将每个模块中的单个前馈神经网络（FFN）替换为包含 4 个专家的混合专家模型（MoE）。配置路由器（Router）并采用 Top-2 路由（top-2 routing）策略。观察在保持活跃参数（active parameters）数量一致的情况下，验证损失的变化情况。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| nanoGPT | “Karpathy 的教程仓库” | 仅包含解码器的极简 Transformer 训练代码，约 300 行代码（LOC）；该领域的标准参考实现。 |
| tinyshakespeare | “标准玩具语料库” | 约 1.1 MB 的文本数据；自 2015 年以来，几乎所有字符级语言模型（character-LM）教程都使用它。 |
| Tied embeddings（权重共享嵌入） | “共享输入/输出矩阵” | 语言模型（LM）输出头的权重等于词元嵌入矩阵的转置；可节省参数量并提升模型质量。 |
| bf16 autocast（bf16 自动混合精度） | “训练精度技巧” | 前向/反向传播使用 bf16 计算，优化器状态保持 fp32；自 2021 年起的行业标准做法。 |
| Gradient clipping（梯度裁剪） | “防止梯度爆炸” | 将全局梯度范数（gradient norm）限制在 1.0 以内；防止训练过程崩溃。 |
| Cosine LR schedule（余弦学习率调度） | “2020 年后的默认配置” | 学习率（LR）先线性上升（预热 warmup），随后按余弦曲线衰减至峰值的 10%。 |
| MFU（模型浮点运算利用率） | “模型 FLOP 利用率” | 实际达到的浮点运算次数（FLOPs）/ 理论峰值；在 2026 年，密集模型（dense）达到 40%、MoE 模型达到 30% 即属优秀水平。 |
| Val loss（验证损失） | “保留集损失” | 在模型从未见过的数据上计算的交叉熵（cross-entropy）；用于检测过拟合（overfitting）。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 经典的逐行注释实现。