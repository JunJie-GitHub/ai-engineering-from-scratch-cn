# 开源模型：架构详解

> 在第 04 课中，你从零开始构建了 GPT-2 Small。2026 年的前沿开源模型属于同一家族，仅包含五到六项具体改动。采用均方根层归一化 (RMSNorm) 替代层归一化 (LayerNorm)。采用 SwiGLU 激活函数替代 GELU。采用旋转位置编码 (RoPE) 替代可学习位置编码。采用分组查询注意力 (GQA) 或多头潜在注意力 (MLA) 替代完整的多头注意力 (MHA)。在大规模场景下采用混合专家模型 (MoE)。你已掌握的数学知识足以覆盖其中 95% 的内容。本课将并排解读 Llama 3、DeepSeek-V3、Mixtral、Qwen 和 Gemma，并明确指出每个架构发生分化的具体位置。

**类型：** 学习
**语言：** Python (标准库)
**前置要求：** 第 10 阶段，第 04、05、12 课（预训练 (Pre-training)、模型扩展 (Scaling)、推理 (Inference)）
**时长：** 约 45 分钟

## 学习目标

- 阅读 Llama 3、Mistral、Mixtral、Gemma 2、Qwen 2.5 和 DeepSeek-V3 的 config.json 文件，并解释其中每个字段的含义
- 指出每个模型相较于 GPT-2 Small 所做的具体架构改动，并从第一性原理出发论证其合理性
- 仅凭配置文件即可计算任意开源模型的参数量、键值缓存 (KV Cache) 大小以及激活内存 (Activation Memory) 占用
- 在给定延迟、内存和性能约束的条件下，为部署目标选择合适的开源模型

## 问题背景

在第 04 课中，你仅用 350 行 NumPy 代码就构建了一个 GPT-2 结构的模型。而 Llama 3 405B 却附带了一份长达 200 页的技术报告。你的直觉可能会认为这两者是完全不同的物种。但事实并非如此。那 200 页内容描述的其实是同一个基础架构，仅包含五到六项经过充分论证的改进，以及上千条关于模型扩展 (Scaling) 的工程实现细节。其核心骨架——词嵌入层 (Embedding)、Transformer 模块 (Transformer Blocks)、注意力机制 (Attention)、多层感知机 (MLP)、归一化层 (Normalization) 以及输出头 (Head)——始终保持不变。

本课本质上是一份差异对比 (Diff)。针对每个主流开源模型家族，我们将精确列出其相较于 GPT-2 的改动内容、改动原因以及带来的计算开销。学完本课后，你将能够阅读全新的模型卡片 (Model Card)，并在脑海中将其自动映射回 GPT-2 基准架构。

其实际价值在于，当 Meta 发布 Llama 5 或 DeepSeek 发布 V4 时，你无需重新建立认知框架。你只需查看配置文件，识别出哪些已知配置项 (Knobs) 发生了调整，便能清楚其对下游任务的影响。2026 年的模型架构本质上是一个有限的工具箱，每个新模型只是从中选取了不同的子集组合。

## 核心概念

### 不变的核心 (Invariant Core)

所有自回归开源模型 (autoregressive open models) 共享以下结构：

- 词元嵌入矩阵 (Token embedding matrix)（`vocab_size` x `hidden_dim`）。
- N 个解码器块 (decoder blocks) 的堆叠：归一化 (norm)、自注意力 (self-attention)、残差连接 (residual)、归一化、多层感知机 (MLP)、残差连接。
- 最终归一化层与线性投影头 (linear head)，输出维度为 `vocab_size`（通常与嵌入层权重共享）。
- 因果掩码 (causal mask) 与下一个词元的交叉熵损失 (next-token cross-entropy loss)。

这就是基本架构。其余部分都是可调节的“旋钮”（设计选择）。

### 真正起作用的六个“旋钮”

纵观 2024-2026 年的所有前沿开源模型，以下六项设计选择被反复采用：

1. **归一化 (Normalization)**：LayerNorm -> RMSNorm。
2. **位置编码 (Positional encoding)**：可学习绝对位置编码 -> 旋转位置编码 (Rotary Position Embedding, RoPE)（及其变体：YaRN、NTK）。
3. **激活函数 (Activation)**：GELU -> SwiGLU（或 GeGLU）。
4. **注意力头共享 (Attention head sharing)**：MHA -> GQA -> MQA -> MLA。
5. **密集与稀疏 MLP**：密集 (Dense) -> 混合专家模型 (Mixture-of-Experts, MoE)。
6. **前置归一化 (Pre-norm) 位置**：保留 Pre-norm。Post-norm 已被淘汰。

其他所有因素（学习率调度、数据混合比例、批次大小、上下文长度）都属于训练配置，而非架构本身。核心就是这六个旋钮。

### 旋钮 1：RMSNorm

LayerNorm 会减去均值、除以标准差，再进行缩放和平移。RMSNorm 仅保留缩放操作：

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

无需减去均值，无偏置项。每个词元减少一次矩阵乘法运算。Zhang 和 Sennrich（2019）指出，它在机器翻译任务上能达到与 LayerNorm 相当的效果，且速度快 10%。如今所有现代开源模型都在使用它。

成本：零。收益：吞吐量小幅提升，代码更简洁。

### 旋钮 2：RoPE

GPT-2 中的可学习位置嵌入是一个包含 1024 个槽位的查找表。当上下文长度达到 1025 时，就会超出表格范围。模型无法外推至超出其训练长度的位置。

旋转位置编码 (Rotary Position Embedding, RoPE, Su et al. 2021) 通过在注意力点积之前成对旋转每个查询 (Q) 和键 (K) 向量来注入位置信息。旋转角度是位置的确定性函数，因此无需学习参数，也不存在“超出范围”的问题。借助缩放技巧（如 NTK 感知插值、YaRN），在 8k 上下文上训练的模型可以在推理时扩展至 128k，且精度损失较小。

q_rotated = rotate(q, angle(pos))
k_rotated = rotate(k, angle(pos))
score = q_rotated . k_rotated

所有 Llama、Mistral、Qwen、DeepSeek 和 Gemma 模型均采用 RoPE。Gemma 2 使用了混合方案（大部分层使用 RoPE，其余层使用局部滑动窗口注意力）。

### 旋钮 3：SwiGLU

GPT-2 的 MLP 结构为 `x -> gelu(xW1 + b1) -> (...)W2 + b2`。SwiGLU（Shazeer 2020）用门控乘积替换了激活函数：

SwiGLU(x) = (xW1) * sigmoid(xW1) * xV

采用两个并行投影而非一个，并由 Swish 激活函数进行门控。经验表明，它在单位参数困惑度 (perplexity) 上表现更强。Llama 2 率先采用后，业界纷纷跟进。MLP 的隐藏层维度通常经过调整，以使总参数量与原始密集 MLP 保持一致：如果 GPT-2 使用 `ff_dim = 4 * hidden`，那么 SwiGLU 则使用 `ff_dim = (2/3) * 4 * hidden = 8/3 * hidden`。

### 旋钮 4：注意力头共享

GPT-2 使用了**多头注意力 (Multi-Head Attention, MHA)**：每个注意力头都有独立的 Q、K、V 投影。

**多查询注意力 (Multi-Query Attention, MQA, Shazeer 2019)** 在所有头之间共享一个 K 和一个 V。KV 缓存 (KV cache) 大小缩减为原来的 `1/num_heads`，在典型模型上可减少 12 到 32 倍。在困难基准测试上精度略有下降。

**分组查询注意力 (Grouped-Query Attention, GQA, Ainslie et al. 2023)** 是折中方案：G 组 Q 头共享一个 K 和一个 V。Llama 3 8B 采用 GQA，包含 32 个 Q 头和 8 个 KV 头（G=8），因此 KV 缓存相比完整 MHA 缩小了 4 倍。

**多头潜在注意力 (Multi-Head Latent Attention, MLA, DeepSeek 2024)** 将 K 和 V 压缩为共享的低秩潜在表示，再按头投影回原始维度。在保持每个头表达能力的同时进一步缩减了 KV 缓存。DeepSeek-V2 和 V3 的长上下文性能正是依赖于此。

| 方案 | KV 头数 | KV 缓存 | 精度 |
|--------|----------|----------|----------|
| MHA    | num_heads | 完整 | 最佳 |
| GQA    | num_groups (G < num_heads) | 缩减为 num_heads / G | 接近 MHA |
| MQA    | 1 | 缩减为 num_heads 分之一 | 轻微下降 |
| MLA    | 潜在表示，按头解压缩 | 小于 MQA | 接近 MHA |

对于参数量超过约 13B 的任何模型，GQA 或 MLA 实际上已成为强制要求。在大规模下使用完整 MHA 会导致 KV 缓存灾难。

### 旋钮 5：混合专家模型 (Mixture of Experts, MoE)

密集 MLP 会对每个词元激活其所有参数。MoE MLP 在每个块中包含 K 个专家，并通过路由网络 (router) 为每个词元选择 top-k 个专家（通常为 top-2）。只有被选中专家的权重会参与该词元的前向传播。

router_logits = xW_r
indices, weights = top_k(router_logits, k=2)
output = sum_i weights[i] * expert[indices[i]](x)

其优势在于：你可以拥有 64 个 7B 大小的专家（总参数量巨大），但每个词元仅运行其中 2 个（因此单词元计算量与密集 7B 模型相当）。Mixtral 8x7B 总参数量为 47B，但每个词元仅激活 13B。DeepSeek-V3 总参数量为 671B，但每个词元仅激活 37B。

graph LR
    I["Token hidden state"] --> R["Router\n(linear -> softmax)"]
    R --> T["Top-k selection"]
    T --> E1["Expert 1\n(MLP)"]
    T --> E2["Expert 2\n(MLP)"]
    T --> EN["Expert 64\n(MLP, unused)"]
    E1 --> S["Weighted sum"]
    E2 --> S
    S --> O["Output"]

    style EN fill:#eeeeee,stroke:#999,color:#999
    style E1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style E2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style R fill:#1a1a2e,stroke:#e94560,color:#fff

优点：计算量相同，参数量更多，模型容量更大。缺点：专家权重仍需存储在显存中（因此部署时所需 VRAM 高于同等规模的密集模型），路由器的负载均衡难以实现，且在模型对齐阶段微调路由器本身就是一个独立的研究方向。

### 旋钮 6：保留前置归一化 (Pre-norm)

原始 Transformer 在每个子层之后应用层归一化。自 GPT-2 以来的所有开源模型都将其移至每个子层*之前*。在深层网络中，Pre-norm 的训练难度严格更低。这一点已无争议。

### 模型逐一对比

下表将所有内容具体化。

| 模型 | 年份 | 总参数量 | 激活参数量 | 归一化 | 激活函数 | 位置编码 | 注意力机制 | MoE | 上下文长度 |
|-------|------|-------------|---------------|------|-----------|----------|-----------|-----|---------|
| GPT-2 Small | 2019 | 124M | 124M | LayerNorm | GELU | 可学习 | MHA (12 heads) | 否 | 1k |
| Llama 3 8B | 2024 | 8B | 8B | RMSNorm | SwiGLU | RoPE | GQA (32/8) | 否 | 128k |
| Llama 3 70B | 2024 | 70B | 70B | RMSNorm | SwiGLU | RoPE | GQA (64/8) | 否 | 128k |
| Llama 3 405B | 2024 | 405B | 405B | RMSNorm | SwiGLU | RoPE | GQA (128/16) | 否 | 128k |
| Mistral 7B | 2023 | 7.2B | 7.2B | RMSNorm | SwiGLU | RoPE | GQA | 否 | 32k |
| Mixtral 8x7B | 2023 | 47B | 13B | RMSNorm | SwiGLU | RoPE | GQA | 是 (8 experts, top-2) | 32k |
| Gemma 2 9B | 2024 | 9B | 9B | RMSNorm (pre+post) | GeGLU | RoPE + sliding | GQA | 否 | 8k |
| Qwen 2.5 72B | 2024 | 72B | 72B | RMSNorm | SwiGLU | RoPE (YaRN) | GQA (64/8) | 否 | 128k |
| DeepSeek V2 236B | 2024 | 236B | 21B | RMSNorm | SwiGLU | RoPE | MLA | 是 (160 experts, top-6) | 128k |
| DeepSeek V3 | 2024 | 671B | 37B | RMSNorm | SwiGLU | RoPE | MLA | 是 (256 experts, top-8) | 128k |

纵向扫描各列：RMSNorm 已成标配。SwiGLU 或其变体 GeGLU 已成标配。RoPE 已成标配。在 7B 以上模型中，GQA 已成标配（除非被 MLA 取代）。MoE 则是顶尖模型拉开差距的关键。

### 解读 config.json

Llama 3 8B 配置：

{
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 131072,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-5,
  "vocab_size": 128256
}

每个字段都对应你已实现过的组件。

- `hidden_size`：嵌入维度。
- `intermediate_size`：MLP 隐藏层维度（3.5 倍 hidden —— SwiGLU 的数学推导）。
- `num_hidden_layers`：堆叠层数。
- `num_attention_heads`：Q 头数量。
- `num_key_value_heads`：KV 头数量（用于 GQA）。
- `max_position_embeddings`：训练上下文长度。
- `rope_theta`：RoPE 基础频率。Meta 将其从默认的 10k 提升至 500k，以支持长上下文外推。
- `rms_norm_eps`：数值稳定性参数。
- `vocab_size`：词表大小。

仅凭这些字段即可计算总参数量、KV 缓存大小和峰值激活内存。具体公式请参见 `code/main.py`。

### 激活内存预算

当参数量达到数十亿级别时，激活值 (activations) 将主导训练内存。预训练的经验法则（结合梯度检查点技术 gradient checkpointing）如下：

activation_mem ~ batch_size * seq_len * hidden_size * num_layers * bytes_per_element

以 Llama 3 8B 为例，batch size 为 1，序列长度 8192，BF16 精度，32 层，hidden 4096：使用检查点时仅激活值就需约 8 GB，不使用则需 40 GB。这正是 Flash-Attention 和 Ring-Attention 至关重要的原因——它们重构了注意力计算流程，使激活值能够适配显存。

### KV 缓存预算

在最大上下文长度下进行推理时：

kv_cache = 2 * num_layers * num_kv_heads * head_dim * max_seq_len * bytes_per_element

Llama 3 8B 在 128k 上下文、BF16 精度下，`head_dim = hidden / num_heads = 128`：
`2 * 32 * 8 * 128 * 131072 * 2 = 17.2 GB` 每条序列。

8B 模型权重在 BF16 下为 16 GB。单条 128k 序列的 KV 缓存甚至超过了模型权重本身。正是这种内存压力推动了 GQA、MLA 以及 KV 缓存量化 (KV cache quantization) 的研究。

### 各模型的适用场景

- **单张 80GB GPU，无 MoE**：Llama 3 8B、Mistral 7B、Gemma 2 9B。部署简单，工具链完善。
- **单节点（8x80GB），大显存容量**：Llama 3 70B、Qwen 2.5 72B。密集架构开源模型中的最强能力。
- **追求最强开源能力，接受 MoE 复杂度**：DeepSeek V3、Mixtral 8x22B。单位激活 FLOP 下的最佳性能。
- **长上下文需求**：Llama 3（借助 RoPE 缩放支持 128k）、DeepSeek（MLA 架构优势）。
- **低延迟部署**：Gemma 2 9B（滑动窗口机制大幅削减长上下文计算量）。

## 构建

本课程的代码实现了一个参数量计算器。给定任意 `config.json`，它会按组件输出参数量、最大上下文长度下的键值缓存（KV Cache）大小、SwiGLU 多层感知机（SwiGLU MLP）比例，并对架构类型给出简短评估（密集架构（Dense）/ 分组查询注意力（GQA）/ 多头潜在注意力（MLA）/ 混合专家模型（MoE））。

config = {
    "hidden_size": 4096, "intermediate_size": 14336,
    "num_hidden_layers": 32, "num_attention_heads": 32,
    "num_key_value_heads": 8, "vocab_size": 128256,
    "max_position_embeddings": 131072,
}

该脚本逐字段解析架构配置，计算词嵌入（Embedding）、注意力机制（含 GQA 缩减）、多层感知机（含 SwiGLU 扩展）、层归一化（LayerNorm）以及输出头（Head）的参数量。随后，它会计算指定上下文长度下的 KV 缓存大小，并打印汇总信息。

具体实现请参见 `code/main.py`。

## 使用

使用脚本内置的 Llama 3 8B、Mistral 7B、Mixtral 8x7B 和 DeepSeek V3 配置文件运行该计算器，并对比各项参数明细。请注意，MoE 模型的总参数量远超密集模型，但其激活参数量（Active Parameter Count）通常更小。同时请注意，尽管 DeepSeek V3 的总参数量更多，但其 KV 缓存却小于 Llama 3 405B——这正是 MLA 架构发挥作用的体现。

接着，将你本地任意模型的配置文件输入其中，阅读汇总结果，并判断其是否适配你的 GPU。

## 部署交付

本课程将生成 `outputs/skill-open-model-picker.md` 文件。给定部署目标（GPU 类型、显存容量、上下文长度、延迟预算）和任务画像（对话、代码、推理、长上下文），它会推荐一款开源模型、第 11 课介绍的量化方案（Quantization Scheme）以及第 12 课的推理栈（Inference Stack），并针对六大架构调节参数（Architectural Knobs）给出明确的推理依据。

## 练习

1. 从 HuggingFace 读取 Qwen 2.5 72B 的配置文件。从零开始计算总参数量。将其与 HF 官方报告的数值进行对比，并找出差异来源（如注意力头维度取整、KV 共享因子等）。

2. DeepSeek V3 采用 256 个专家网络并配合 Top-8 路由机制。计算激活专家数与总专家数的比例，并与 Mixtral 8x7B 的 Top-2/8 进行对比。从稀疏（25%）转向“更密集的稀疏”（3%）对每浮点运算（FLOP）的容量意味着什么？

3. 计算 Llama 3 405B 在 128k 上下文长度下，分别采用 FP8 和 BF16 精度时的 KV 缓存大小。在 FP8 精度下，其大小应为 BF16 的一半。在单个 8xH100 节点（每张卡 80GB，总计 640GB，需扣除权重内存）上，你能同时服务多少个并行序列？

4. Gemma 2 交替使用全注意力层与滑动窗口注意力层。请推导当一半层使用 4096 词元（Token）的滑动窗口而非全上下文时，KV 缓存的计算公式。在总上下文长度为 8k 的情况下，这能节省多少内存？

5. 寻找一个在本课程编写后发布的最新前沿开源模型。识别它采用了六大架构调节参数中的哪些，以及是否引入了第七个参数。每当新架构发布时，课程内容都会显得过时——我们的目标是让你在不重构底层心智模型（Mental Model）的前提下，持续更新你的评估表格。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 均方根层归一化 (RMSNorm) | “去掉了均值计算的层归一化 (LayerNorm)” | 仅使用均方根进行归一化，并带有可学习的缩放参数——计算成本更低，且效果与层归一化 (LayerNorm) 相当。 |
| 旋转位置编码 (RoPE) | “旋转位置” | 将查询 (Q) 和键 (K) 向量按二维配对，根据位置旋转相应角度——结合缩放技巧可外推至超出训练长度的上下文。 |
| SwiGLU 门控线性单元 (SwiGLU) | “新一代多层感知机 (MLP) 激活函数” | 结合 Swish 的门控线性单元：`(xW1) * sigmoid(xW1) * xV`——已成为 2024 年及以后所有开源模型的标准配置。 |
| 分组查询注意力 (GQA) | “折中方案注意力机制” | 分组查询注意力：G 组查询头共享一个键头和一个值头——在不损失多查询注意力 (MQA) 精度的前提下，有效缩减键值缓存 (KV cache)。 |
| 多头潜在注意力 (MLA) | “DeepSeek 的注意力机制” | 多头潜在注意力：将键/值压缩至共享的低秩潜在空间，再按头解压缩——为大规模模型提供最小的键值缓存 (KV cache)。 |
| 混合专家模型 (MoE) | “稀疏专家” | 混合专家模型：每个模块包含 N 个多层感知机 (MLP)，路由器为每个词元 (token) 选择 top-k 个专家——总参数量巨大，但激活参数量很小。 |
| Top-k 路由 (Top-k routing) | “每个词元 (token) 挑选 k 个专家” | 路由器为每个专家计算得分并激活得分最高的 k 个——典型的 k 值范围为 2（如 Mixtral）到 8（如 DeepSeek）。 |
| YaRN 上下文扩展 (YaRN) | “拉伸版 RoPE” | 另一种 RoPE 扩展方法——通过插值旋转角度，在推理阶段将上下文窗口从 8k 扩展至 128k 以上。 |
| 滑动窗口注意力 (Sliding-window attention) | “不必关注所有内容” | 每个词元 (token) 仅关注其前 W 个词元——将每个词元的注意力计算成本限制在 O(W)，已应用于 Gemma 2 和早期 Mistral 模型。 |
| 激活参数 (Active params) | “每个词元 (token) 实际运行的参数” | 针对 MoE 模型，指每个词元在前向传播中实际调用的参数量（远小于总参数量）——直接决定每个词元的浮点运算次数 (FLOPs)。 |

## 延伸阅读

- [Dubey 等人, 2024 -- "The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783) -- 密集架构 Llama 3 系列模型的架构与训练参考指南
- [DeepSeek-AI, 2024 -- "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437) -- 结合 MLA、无辅助损失负载均衡以及 6710 亿参数的 MoE 架构
- [Jiang 等人, 2024 -- "Mixtral of Experts"](https://arxiv.org/abs/2401.04088) -- MoE 开源模型的奠基性论文
- [Su 等人, 2021 -- "RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864) -- RoPE 的原始论文
- [Shazeer, 2020 -- "GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) -- 介绍 SwiGLU、GeGLU 及其变体
- [Ainslie 等人, 2023 -- "GQA: Training Generalized Multi-Query Transformer Models"](https://arxiv.org/abs/2305.13245) -- GQA 的原始论文
- [Gemma 2 团队, 2024 -- "Gemma 2: Improving Open Language Models at a Practical Size"](https://arxiv.org/abs/2408.00118) -- 采用全注意力与滑动窗口注意力混合架构，以及前置与后置归一化结合
- [Qwen 团队, 2024 -- "Qwen 2.5 Technical Report"](https://arxiv.org/abs/2412.15115) -- YaRN 上下文扩展技术与长上下文训练方案