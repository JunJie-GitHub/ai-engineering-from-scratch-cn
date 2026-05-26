# 为什么选择 Transformer —— RNN 的问题

> 循环神经网络（Recurrent Neural Network, RNN）逐个处理词元（token）。Transformer 则一次性处理所有词元。这一架构上的单一抉择改变了 2017 年后深度学习领域的所有缩放曲线（scaling curve）。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 3 阶段（深度学习核心），第 5 阶段 · 09（序列到序列），第 5 阶段 · 10（注意力机制）
**Time:** 约 45 分钟

## 问题所在

在 2017 年之前，全球所有最先进的序列模型（sequence model）——涵盖语言、翻译和语音领域——无一例外都是循环神经网络。长短期记忆网络（Long Short-Term Memory, LSTM）和门控循环单元（Gated Recurrent Unit, GRU）在长达五年的时间里主导了与 ImageNet 同等地位的翻译基准测试（benchmark）。它们是当时业界唯一可用的工具。

它们存在三个致命缺陷。首先是顺序计算（sequential computation）：这意味着无法沿时间轴进行并行化处理，因为词元 `t+1` 的计算依赖于词元 `t` 的隐藏状态（hidden state）。在一个包含 1,024 个词元的序列中，即便 GPU 每个周期能执行 100 万次浮点运算（floating-point operations），也必须串行执行 1,024 个步骤。在专为并行计算设计的硬件上，训练的实际耗时（wall-clock time）却与序列长度呈线性增长。

其次是梯度消失（vanishing gradient）问题：这意味着向前回溯 50 个词元的信息，在穿过 50 层非线性变换（non-linearities）后早已被严重压缩。门控循环单元（LSTM、GRU）虽然缓解了这种信息挤压，但从未彻底消除它。长距离依赖（long-range dependency）——例如“我去年夏天飞往京都的飞机上读的那本书是……”——在模型中经常无法被正确捕捉。

最后是固定维度的隐藏状态（fixed-width hidden state）：这意味着编码器（encoder）必须在解码器（decoder）处理任何内容之前，将整个源序列压缩进一个单一向量中。无论源序列是 5 个词元还是 500 个词元，这个瓶颈的维度形状始终不变。

2017 年的论文《Attention Is All You Need》提出了一种颠覆性的方案：彻底抛弃循环结构（recurrence）。让序列中的每个位置都能并行地关注其他所有位置。通过一次大规模的矩阵乘法（matrix multiplication）完成训练，而非 1,024 次顺序计算。

到 2026 年，这一架构已主导了所有模态（modality）。语言领域（GPT-5、Claude 4、Llama 4）、视觉领域（ViT、DINOv2、SAM 3）、音频领域（Whisper）、生物学领域（AlphaFold 3）以及机器人领域（RT-2）。相同的网络模块（block），仅需更换不同的输入数据。

## 核心概念

![RNN 顺序计算与 Transformer 并行注意力机制](../assets/rnn-vs-transformer.svg)

**循环机制成为瓶颈。** 循环神经网络（Recurrent Neural Network, RNN）的计算公式为 `h_t = f(h_{t-1}, x_t)`。每一步都依赖于前一步的结果。在计算出 `h_4` 之前，无法计算 `h_5`。在拥有上万个并行核心的现代 GPU 上，处理长序列时会导致 99% 的芯片算力被闲置浪费。

**注意力机制如同广播。** 自注意力机制（Self-Attention）会同时为每一对 `(i, j)` 计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N 的注意力矩阵可通过一次批量矩阵乘法（Batched MatMul）一次性填充完成。各步骤之间互不依赖。GPU 对此极为友好。

**加速效果并非固定常数。** 它本质上是 `O(N)` 串行深度与 `O(1)` 串行深度之间的差异。在实际应用中，当序列长度 N=512 且硬件配置相当时，Transformer 每个训练轮次（Epoch）的速度比 RNN 快 5 到 10 倍。随着序列长度的增加，这一差距会进一步拉大，直到触及注意力机制 `O(N²)` 的内存墙（Memory Wall）（该问题后来被 Flash Attention 解决——详见第 12 课）。

**Transformer 的代价。** 注意力机制的内存占用按 `O(N²)` 扩展。对于 2K 的上下文长度，这完全不是问题。但对于 128K 的上下文长度，则需要引入滑动窗口（Sliding Window）、RoPE 外推（RoPE Extrapolation）、Flash Attention 分块（Tiling）或线性注意力（Linear Attention）变体。循环机制在时间和内存上均为 `O(N)`；Transformer 则是以内存换时间，再通过并行计算将时间优势赢回来。

**归纳偏置（Inductive Bias）的转变。** RNN 假设数据具有局部性（Locality）与近期性（Recency）。Transformer 则不做任何预设——任意两个词元（Token）之间都可能产生注意力交互。这就是为什么 Transformer 需要更多数据才能训练出良好效果，但一旦数据充足，其扩展能力（Scaling）将远超 RNN。Chinchilla 研究（2022）对此进行了形式化论证：在给定足够多词元的情况下，参数量相同的 Transformer 始终优于 RNN。

## 动手实践

这里不涉及神经网络——我们通过数值模拟核心瓶颈，让你能在自己的笔记本电脑上直观感受到这种性能差距。

### 步骤 1：测量串行深度

参见 `code/main.py`。我们构建了两个函数。一个将序列编码为加法链（串行，类似于循环神经网络 (Recurrent Neural Network, RNN)）。另一个将其编码为并行归约 (Parallel Reduction)（广播 (Broadcast)，类似于注意力机制 (Attention)）。数学原理相同，但依赖图 (Dependency Graph) 不同。

def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent

我们在长度高达 100,000 个元素的序列上对两者进行计时。RNN 版本的时间复杂度为 O(N)，且仅使用单条 CPU 流水线。即使在纯 Python 环境中，当序列长度 ≥ 1,000 时，注意力风格的归约操作也能胜出，因为 Python 的 `sum()` 函数底层由 C 语言实现，迭代过程中避免了每一步的解释器开销。

### 步骤 2：计算理论运算量

两种算法都执行 N 次加法。区别在于*依赖深度 (Dependency Depth)*：在下一个操作开始前，必须按顺序执行多少步操作。RNN 的深度为 N。使用树形归约 (Tree Reduction) 时，注意力机制的深度为 log(N)；使用并行扫描 (Parallel Scan) 时，深度为 1。决定 GPU 运行时间的是深度，而非运算次数。

### 步骤 3：长序列上的经验缩放测试

我们打印了一张计时表，使 O(N) 的性能差距一目了然。在 2026 款的 Mac 笔记本电脑上，长度不足 1,000 的序列运行太快，难以精确测量。而长度为 100,000 的序列则呈现出清晰的线性扫描特征。将这一比例放大到具有等效 12 层长短期记忆网络 (Long Short-Term Memory, LSTM) 的 16,384 词元 (Token) Transformer 模型上，你就能明白为什么在 2016 年，训练的实际耗时 (Wall-clock Time) 会成为一大瓶颈。

## 实际应用

在 2026 年，何时仍应选择 RNN：

| 场景 | 推荐选择 |
|-----------|------|
| 流式推理 (Streaming Inference)，逐词元生成，内存占用恒定 | RNN 或状态空间模型 (State-Space Model, SSM)（如 Mamba、RWKV） |
| 极长序列（>100 万词元），注意力机制内存爆炸 | 线性注意力 (Linear Attention)、Mamba 2、Hyena |
| 无矩阵乘法 (Matrix Multiplication, MatMul) 加速器的边缘设备 | 深度可分离 RNN (Depthwise-separable RNN) 在每瓦特浮点运算次数 (FLOPs/Watt) 上仍占优 |
| 其他所有情况（训练、批量推理、上下文长度达 128K） | Transformer |

像 Mamba 这样的状态空间模型 (SSM) 本质上是经过结构化参数化 (Structured Parameterization) 的 RNN，兼具两者优势：`O(N)` 的扫描内存占用，以及通过选择性扫描 (Selective Scan) 实现的并行训练。它们能以更优的长上下文扩展能力，恢复 Transformer 90% 的性能质量。到 2026 年，大多数前沿实验室都在训练 SSM 与 Transformer 的混合架构模型（例如 Jamba、Samba）——循环机制并未消亡，而是成为了模型的一个组件。

## 发布上线

参见 `outputs/skill-architecture-picker.md`。该技能模块会根据序列长度、吞吐量 (Throughput) 和训练预算约束，为新的序列处理问题挑选合适的架构。对于训练数据量超过 10 亿词元的任务，它必须始终拒绝推荐纯 RNN 架构，除非明确说明其中的权衡取舍 (Trade-off)。

## 练习

1. **简单。** 从 `code/main.py` 中取出 `rnn_style`，将标量隐藏状态（scalar hidden state）替换为长度为 64 的隐藏状态向量。重新进行测量。串行开销（serial overhead）会随着隐藏状态维度（hidden-state dimension）的增加而增长多少？
2. **中等。** 使用纯 Python 实现并行前缀和（parallel prefix-sum，即 Hillis-Steele 扫描算法）。验证其在序列长度为 1024 时，输出的数值结果是否与串行扫描（serial scan）完全一致。统计计算深度（depth）。
3. **困难。** 将注意力风格的归约操作（attention-style reduction）移植到基于 GPU 的 PyTorch 环境中。在序列长度（sequence length）从 64 逐步增加至 65,536 的过程中，分别对两者进行耗时测试。绘制性能曲线并解释其形态特征。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 循环计算（Recurrence） | “RNN 是顺序执行的” | 第 `t` 步的计算依赖于第 `t-1` 步，从而强制沿时间轴进行串行执行。 |
| 串行深度（Serial depth） | “计算图有多深” | 依赖操作的最长链；即使在无限硬件上，它也限制了实际运行时间（wall-clock time）。 |
| 注意力机制（Attention） | “让词元互相查看” | 加权和 `sum_j a_ij v_j`，其中权重 `a_ij` 来源于位置 i 和 j 之间的相似度得分。 |
| 上下文窗口（Context window） | “模型能看到多少内容” | 注意力层能够接收的输入位置数量；二次方内存开销（quadratic memory cost）在此处随规模增长。 |
| 归纳偏置（Inductive bias） | “架构中内置的假设” | 关于数据分布的先验假设；例如 CNN 假设平移不变性（translation invariance），RNN 假设近期信息更重要。 |
| 状态空间模型（State-space model） | “带有代数背景的 RNN” | 通过结构化状态空间矩阵进行参数化的循环结构，以支持并行训练。 |
| 二次方瓶颈（Quadratic bottleneck） | “为什么上下文成本如此之高” | 注意力机制的内存消耗随序列长度呈 `O(N²)` 增长；Flash Attention 仅隐藏了常数项，并未改变其缩放规律。 |

## 扩展阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 这篇论文终结了主流自然语言处理（NLP）中的循环结构。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 注意力机制的诞生之作，最初是作为附加模块集成到 RNN 上的。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) —— LSTM 的原始论文，在此存档备查。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) —— 针对 Transformer 架构的现代循环结构替代方案。