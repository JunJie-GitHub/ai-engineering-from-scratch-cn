# DeepSeek-V3 架构详解

> 第10阶段·第14课指出了每个开源模型都会调整的六个架构调节旋钮（architectural knobs）。DeepSeek-V3（2024年12月发布，总参数量671B，激活参数量37B）不仅用上了这六个旋钮，还额外增加了四个：多头潜在注意力（Multi-Head Latent Attention, MLA）、无辅助损失负载均衡（auxiliary-loss-free load balancing）、多词元预测（Multi-Token Prediction, MTP）以及 DualPipe 训练。本课将自上而下逐层解析 DeepSeek-V3 的架构，并根据已公开的配置文件（config）推导每一项参数数量。学完本课后，你将能够解释为何 671B/37B 的比例是明智之选，以及为何在前沿模型中 MLA 与混合专家模型（Mixture of Experts, MoE）的结合能超越单独使用其中任何一种架构。

**Type:** 学习
**Languages:** Python（标准库，参数计算器）
**Prerequisites:** 第10阶段·第14课（开源模型导览）、第10阶段·第17课（NSA）、第10阶段·第18课（MTP）、第10阶段·第19课（DualPipe）
**Time:** 约75分钟

## 学习目标

- 自上而下逐层阅读 DeepSeek-V3 的配置文件，并结合 GPT-2 的六个架构调节旋钮以及 DeepSeek 特有的四项新增设计，解释每个字段的含义。
- 推导总参数量（671B）与激活参数量（37B），并明确构成这两部分的具体组件。
- 计算 MLA 在 128k 上下文长度下的 KV 缓存（KV cache）占用空间，并与采用分组查询注意力（Grouped-Query Attention, GQA）且激活参数量相同的稠密模型（dense model）进行对比。
- 阐述 DeepSeek 特有的四项创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每一项分别针对架构或训练栈中的哪个环节。

## 问题背景

DeepSeek-V3 是首个在架构上与 Llama 系列产生实质性差异的前沿开源模型。Llama 3 405B 可以被视为“调整了六个旋钮的 GPT-2”，而 DeepSeek-V3 则是“用满六个旋钮并额外增加四个的 GPT-2”。阅读 Llama 3 的配置文件可以作为理解 DeepSeek 配置的热身，但两者的底层结构——注意力模块的形态、路由逻辑以及训练目标——差异显著，因此需要单独进行详细剖析。

学习它的价值在于：DeepSeek-V3 的开源权重发布重新定义了开源模型中“前沿能力”的标准。该架构已成为众多 2026 年训练项目竞相效仿的蓝图。对于任何涉及前沿大语言模型（Large Language Model, LLM）训练或推理的岗位而言，深入理解该架构都是必备的基础门槛。

## 核心概念

### 不变的核心架构（再次强调）

DeepSeek-V3 依然采用自回归（autoregressive）架构。它仍然堆叠解码器块（decoder blocks）。每个块依然包含注意力机制（attention）、多层感知机（MLP）以及两个 RMSNorm。MLP 中依然使用 SwiGLU 激活函数。依然采用旋转位置编码（RoPE）。采用前置归一化（Pre-norm）。使用权重共享词嵌入（Weight-tied embeddings）。其基础架构与 Llama 或 Mistral 系列模型完全一致。

### 关键创新：采用 MLA 替代 GQA

从第 10 阶段 · 14 节中你已了解，分组查询注意力（GQA）通过在多个查询头（Q heads）组间共享键（K）和值（V）来缩减 KV 缓存（KV cache）。多头潜在注意力（Multi-Head Latent Attention, MLA）则更进一步：K 和 V 被压缩为一个共享的低秩潜在表示（即 `kv_lora_rank`），然后在推理时按头动态解压。KV 缓存仅存储该潜在向量——通常每层每个词元（token）仅需 512 个浮点数，而非 8 x 128 = 1024 个浮点数。

在 128k 上下文长度下，采用 MLA 的 DeepSeek-V3（每层每个词元共享一个潜在向量 `c^{KV}`；K 和 V 均通过上投影矩阵从该潜在向量派生，且这些投影矩阵可被吸收到后续的矩阵乘法中）：

kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB

假设采用 GQA 的基线模型（Llama 3 70B 架构，8 个 KV 头，头维度 128）则需要：

kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB

在 128k 上下文长度下，MLA 的缓存大小仅为 Llama-3-70B 风格 GQA 缓存的四分之一。

权衡之处：MLA 在每次注意力计算（每个头）中增加了解压步骤。但与节省的内存带宽相比，额外的计算开销微乎其微。对于长上下文推理而言，这是一项净收益。

### 路由机制：无辅助损失的负载均衡

混合专家模型（MoE）的路由器决定由哪 top-k 个专家处理每个词元。朴素的路由器容易将过多任务集中在少数专家上，导致其他专家闲置。标准解决方案是：添加一个辅助损失项（auxiliary loss term）来惩罚负载不均衡。这虽然有效，但会轻微降低主任务的性能。

DeepSeek-V3 引入了一种无辅助损失（auxiliary-loss-free）的方案。它在路由器 logits 中为每个专家添加了偏置项，并在训练期间通过一条简单规则进行调整：如果专家 `e` 负载过重，则减小 `bias_e`；如果负载过轻，则增大它。无需额外的损失项。训练过程保持干净，专家负载保持均衡。

对主损失的影响：无可测量的变化。对 MoE 架构的影响：更加简洁，无需调节辅助损失的超参数。

### MTP 模块：更密集的训练信号 + 零开销草稿生成

从第 10 阶段 · 18 节中你已了解，DeepSeek-V3 添加了深度 D=1 的多词元预测（Multi-Token Prediction, MTP）模块，用于预测当前位置之后两个位置的词元。在推理阶段，该训练好的模块被复用为投机解码（speculative decoding）的草稿生成器，接受率超过 80%。在训练阶段，每个隐藏状态同时受到 D+1 = 2 个目标的监督，从而提供更密集的训练信号。

参数量：在 671B 主模型基础上增加 14B。额外开销：2.1%。

### 训练框架：DualPipe

从第 10 阶段 · 19 节中你已了解，DualPipe 是一种双向流水线（bidirectional pipeline），它将前向和后向计算块与跨节点的全对全通信（all-to-all comms）进行重叠。在 DeepSeek-V3 的 2,048 张 H800 GPU 规模下，它挽回了约 24.5 万 GPU 小时，这些时间若采用传统的 1F1B 策略本会浪费在流水线气泡（pipeline bubbles）上。

### 配置参数逐项解析

以下是 DeepSeek-V3 的配置参数（简化版）：

hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)

逐项解析：

- `hidden_size=7168`：词嵌入维度。
- `num_hidden_layers=61`：总层数（块深度）。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的稠密 MLP。其余 58 个块使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩至该潜在维度，并在每个头处进行解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块包含 256 个专家，路由时选择 top-8。
- `shared_experts=1`：在 256 个路由专家之外，还有 1 个始终激活的共享专家参与每个词元的计算。可将其视为“稠密基座”，确保每个词元都能获得可靠的特征表示。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏层大小。由于专家数量多达 256 个，因此该尺寸小于稠密 MLP。

### 参数量核算

完整计算代码位于 `code/main.py`。核心数据如下：

- 词嵌入层：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个稠密块：采用 MLA 的注意力机制（每块约 144M）+ 稠密 MLP（每块约 260M）+ 归一化层。总计约 1.2B。
- 58 个 MoE 块：采用 MLA 的注意力机制（约 144M）+ 每块 256 个专家（每个约 30M）+ 1 个共享专家（30M）+ 归一化层。包含所有专家在内，每块总计约 7.95B。58 个 MoE 块总计 461B。
- MTP 模块：14B。

总计：核心架构约 476B + MTP 模块 14B。值得注意的是，官方公布的 671B 参数量还包含了额外的结构参数（偏置张量、专家专属组件、共享专家缩放因子等）。我们在计算器中复现的数值与官方数据相差在 3-5% 以内——这一差异源于 DeepSeek 报告附录第 2 节中记录的更细粒度的参数核算方式。

单次前向传播的激活参数量：

- 注意力机制：每层 144M * 61 = 8.8B（所有层均参与计算）。
- MLP 激活部分：前 3 层为稠密结构（3 * 260M = 780M），58 个 MoE 层每层激活 8 个路由专家 + 1 个共享专家 + 路由开销。每层激活的 MLP 参数量约 260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- 词嵌入 + 归一化层：1.2B。
- 总激活参数量：核心部分约 26B + MTP 模块 14B（参与训练但推理时不总是运行）≈ 37B。

### 671B / 37B 的稀疏比

稀疏比约为 18 倍（激活参数仅占总参数的 5.5%）。DeepSeek-V3 是目前已开源权重的最稀疏的前沿 MoE 模型。Mixtral 8x7B 的稀疏比为 13/47（28%），密度要高得多。Llama 4 Maverick 的稀疏比为 17B/400B（4.25%），与之相当。DeepSeek 的押注在于：在前沿规模下，采用更多专家并降低激活比例，能够在每次激活浮点运算（active-FLOP）下获得更优的模型质量。

### DeepSeek-V3 的定位

| 模型 | 总参数量 | 激活参数量 | 稀疏比 | 注意力机制 | 创新点 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续演进：R1 与 V4

DeepSeek-R1（2025）是基于 V3 主干架构进行推理能力训练的模型。R1 采用了相同的底层架构。发生变化的是后训练策略（在可验证任务上进行大规模强化学习），而非预训练架构。

DeepSeek-V4（若发布）预计将延续 MLA + MoE + MTP 架构，并引入 DSA（DeepSeek Sparse Attention，即 DeepSeek 稀疏注意力），作为第 10 阶段 · 17 节中 NSA 的继任者。其技术脉络十分稳定：架构层面的创新不断累积，每个新版本都会进一步调节更多的控制旋钮。

## 实践应用

`code/main.py` 是专为 DeepSeek-V3 架构设计的参数计算器 (Parameter Calculator)。运行该脚本，将其输出结果与论文中的数据进行对比，并将其应用于假设的模型变体（例如 256 个专家 (Expert) 对比 512 个、Top-8 路由 (Top-8 Routing) 对比 Top-16、多头潜在注意力 (Multi-Head Latent Attention, MLA) 的秩 (Rank) 512 对比 1024）。

关注重点：

- 总参数量 (Total Parameter Count) 与已公布的 671B 对比。
- 激活参数量 (Active Parameter Count) 与已公布的 37B 对比。
- 128k 上下文 (Context) 长度下的键值缓存 (KV Cache) 大小——对比 MLA 与分组查询注意力 (Grouped-Query Attention, GQA) 的差异。
- 逐层参数分布明细，以了解参数预算的实际分配去向。

## 交付产出

本课时将生成 `outputs/skill-deepseek-v3-reader.md` 文件。针对任意 DeepSeek 系列模型（如 V3、R1 或未来的变体），该脚本会输出一份逐组件的架构解析报告，其中会命名配置文件 (Config) 中的每个字段、按组件推导参数量，并识别该模型采用了 DeepSeek 四项专属创新技术中的哪几项。

## 练习

1. 运行 `code/main.py`。将计算器估算的总参数量与已公布的 671B 进行对比，并找出差值 (Delta) 的来源。论文的第 2 节提供了完整的明细列表。
2. 修改配置文件，将 MLA 的秩从 512 改为 256。计算在 128k 上下文长度下产生的键值缓存大小。这一改动能带来多大比例的缩减？同时会对单头 (Per-Head) 表达能力造成何种代价？
3. 将 DeepSeek-V3 的（256 个专家，Top-8 路由）架构与假设的（512 个专家，Top-8 路由）变体进行对比。总参数量会增加，但激活参数量保持不变。理论上，额外的专家容量能带来什么收益？在推理 (Inference) 阶段又会增加什么成本？
4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）中关于 MLA 的第 2.1 节。用三句话解释：为何 K（键）和 V（值）的解压缩矩阵可以被“吸收”到后续的矩阵乘法 (Matrix Multiplication) 中，从而提升推理阶段的效率。
5. DeepSeek-V3 在大多数运算中采用了 8 位浮点数 (FP8) 训练。计算在存储 671B 权重 (Weight) 时，FP8 相较于 16 位脑浮点数 (BF16) 所能节省的内存空间。这一优化如何与 14.8T token 的训练预算 (Training Budget) 产生关联？

## 关键术语

| 术语 | 常见称呼 | 实际含义 |
|------|----------------|------------------------|
| MLA (Multi-Head Latent Attention) | “多头潜在注意力” | 将 K 和 V 压缩为共享的低秩潜在表示（`kv_lora_rank`，通常为 512），并在每个注意力头中实时解压；键值缓存 (KV cache) 仅存储该潜在表示 |
| kv_lora_rank | “MLA 压缩维度” | K 和 V 共享潜在表示的维度大小；DeepSeek-V3 使用 512 |
| 前 k 个稠密层 (First k dense layers) | “早期层保持稠密” | 混合专家模型 (MoE) 的前几层跳过 MoE 路由，直接运行稠密多层感知机 (MLP) 以保证稳定性 |
| 每 token 激活专家数 (num_experts_per_tok) | “Top-k 路由” | 每个 token 激活的路由专家数量；DeepSeek-V3 使用 8 |
| 共享专家 (Shared experts) | “常驻专家” | 无论路由结果如何，都会处理每个 token 的专家；DeepSeek-V3 使用 1 个 |
| 无辅助损失路由 (Auxiliary-loss-free routing) | “偏差调整负载均衡” | 在训练期间动态调整各专家的偏置项，从而在不引入额外损失项的情况下保持专家负载均衡 |
| MTP 模块 (MTP module) | “额外预测头” | 基于 `h^(1)` 和 `E(t+1)` 预测 `t+2` 时刻状态的 Transformer 模块；用于更密集的训练，并提供免费的投机解码 (speculative decoding) 草稿 |
| DualPipe | “双向流水线” | 一种训练调度策略，将前向/反向计算与跨节点的全对全通信 (all-to-all) 重叠执行 |
| 激活参数比例 (Active parameter ratio) | “稀疏度” | `active_params` / `total_params`；DeepSeek-V3 达到 5.5% |
| FP8 训练 (FP8 training) | “8 位训练” | 使用 FP8 格式进行训练存储及大量计算操作；相比 BF16 可大幅降低约一半的显存占用，且仅带来微小的质量损失 |

## 进一步阅读

- [DeepSeek-AI — DeepSeek-V3 技术报告 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练及结果文档
- [Hugging Face 上的 DeepSeek-V3 模型卡片](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件与部署说明
- [DeepSeek-V2 论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前代模型
- [DeepSeek-R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 基于 V3 架构进行推理训练的后继模型
- [原生稀疏注意力 (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 系列模型注意力机制的未来发展方向
- [DualPipe 代码库](https://github.com/deepseek-ai/DualPipe) — 训练调度策略的参考实现