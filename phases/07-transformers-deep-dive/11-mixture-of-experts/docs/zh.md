# 混合专家模型 (Mixture of Experts, MoE)

> 一个 70B 的稠密 Transformer (Dense Transformer) 会为每个词元 (Token) 激活全部参数。而一个 671B 的 MoE 模型每个词元仅激活 37B 参数，却在所有基准测试 (Benchmark) 中全面胜出。稀疏性 (Sparsity) 是本十年最重要的模型扩展 (Scaling) 理念。

**类型:** 构建
**语言:** Python
**前置要求:** 第 7 阶段 · 05（完整 Transformer），第 7 阶段 · 07（GPT）
**耗时:** 约 45 分钟

## 核心问题

在推理阶段，稠密 Transformer 的浮点运算次数 (FLOPs) 等于其参数量（前向传播需乘以 2）。若扩大稠密模型规模，每个词元都需承担全额计算开销。到 2024 年，前沿模型已触及算力墙 (Compute Wall)：要想实现显著的智能跃升，每个词元所需的 FLOPs 必须呈指数级增长。

混合专家模型打破了这一绑定关系。它将每个前馈神经网络 (FFN) 替换为 `E` 个独立的专家网络 (Expert)，并引入一个路由模块 (Router) 为每个词元挑选 `k` 个专家。总参数量 = `E × FFN_size`。每个词元激活的参数量 = `k × FFN_size`。典型的 2026 年配置为：`E=256`，`k=8`。存储规模随 `E` 扩展，计算规模随 `k` 扩展。

2026 年的前沿模型几乎清一色采用 MoE 架构：DeepSeek-V3（总计 671B / 激活 37B）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，排名前十的开源模型全部为 MoE 架构。

## 核心概念

![混合专家（MoE）层：路由器（router）为每个词元（token）选择 E 个专家中的 k 个](../assets/moe.svg)

### 前馈神经网络（FFN）替换

稠密 Transformer 模块：

h = x + attn(norm(x))
h = h + FFN(norm(h))

MoE 模块：

h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )

每个专家（expert）都是独立的前馈神经网络（FFN）（通常采用 SwiGLU 结构）。路由器（router）为单个线性层。每个 token 会自主选择 `k` 个专家，并获得其输出的门控混合（gated mixture）结果。

### 负载均衡（load-balancing）问题

如果路由器将 90% 的 token 都路由给专家 3，其他专家就会“饿死”（得不到使用）。业界尝试过三种解决方案：

1. **辅助负载均衡损失（auxiliary load-balancing loss）**（Switch Transformer、Mixtral）。添加与专家使用率方差成正比的惩罚项。该方法有效，但会引入额外的超参数和第二个梯度信号。
2. **专家容量与词元丢弃（expert capacity + token dropping）**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 将跳过该层。这会损害模型质量。
3. **无辅助损失负载均衡（auxiliary-loss-free balancing）**（DeepSeek-V3）。为每个专家添加一个可学习的偏置项（bias），用于偏移路由器的 top-k 选择。该偏置项在训练损失之外进行更新，不会对主目标函数施加惩罚。这是 2024 年的一项重大突破。

DeepSeek-V3 的具体做法：在每个训练步骤后，检查每个专家的使用率是高于还是低于目标值。将偏置项微调 `±γ`。选择过程使用 `scores + bias`。而用于门控的专家概率则保持原始的 `scores` 不变。这实现了路由决策与特征表达的解耦。

### 共享专家（shared experts）

DeepSeek-V2/V3 还将专家划分为*共享专家（shared）*和*路由专家（routed）*。每个 token 都会经过所有共享专家。路由专家则通过 top-k 机制选取。共享专家负责捕获通用知识，而路由专家专注于特定领域。V3 架构包含 1 个共享专家，以及从 256 个路由专家中选取的 top-8。

### 细粒度专家（fine-grained experts）

经典 MoE 架构（GShard、Switch）：每个专家的宽度与完整的 FFN 相当。`E`（专家总数）较小（8–64），`k`（激活专家数）也较小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（约为 FFN 大小的 1/8）。`E` 较大（256+），`k` 也更大（8+）。总参数量相同，但组合数量呈指数级增长。每个 token 可能的“专家”组合高达 `C(256, 8) = 400 万亿` 种。模型质量显著提升，而延迟保持不变。

### 成本分析（cost profile）

每个 token、每层的计算情况：

| 配置 | 每个 token 激活参数量 | 总参数量 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 在几乎所有基准测试中均优于 Llama 3 70B（稠密模型），同时每个 token 的**激活浮点运算次数（FLOPs）更少**。参数量越多 = 知识储备越丰富。激活 FLOPs 越多 = 每个 token 的计算量越大。MoE 架构将这两者成功解耦。

### 瓶颈：显存（memory）

无论哪些专家被激活，所有专家都必须驻留在 GPU 上。一个 671B 参数的模型仅 fp16 权重就需要约 1.3 TB 的显存（VRAM）。前沿 MoE 模型的部署需要采用专家并行（expert parallelism）——将专家分片（shard）到多个 GPU 上，并在网络中路由 token。此时，延迟主要受限于全对全通信（all-to-all communication），而非矩阵乘法（matmul）计算。

## 构建它

参见 `code/main.py`。这是一个仅使用标准库实现的紧凑混合专家 (Mixture of Experts, MoE) 层，包含：

- `n_experts=8` 个类 SwiGLU (Swish-Gated Linear Unit) 专家（为便于演示，每个专家仅包含一个线性层）
- top-k=2 路由 (Routing)
- 经 Softmax 归一化的门控权重 (Gating Weights)
- 通过每个专家的偏置项实现无辅助损失 (Auxiliary Loss) 的负载均衡

### 步骤 1：路由模块 (Router)

def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates

偏置项仅影响专家选择，而不影响门控权重。这正是 DeepSeek-V3 的巧妙之处——通过偏置项修正负载不均衡，同时不会干扰模型的预测结果。

### 步骤 2：让 100 个词元 (Token) 通过路由模块

记录各个专家的激活频率。若不引入偏置项，专家使用率会出现严重倾斜。通过偏置更新循环（对过度使用的专家施加 `-γ`，对使用不足的专家施加 `+γ`），经过数次迭代后，使用率将收敛至均匀分布。

### 步骤 3：参数量对比

打印 MoE 配置对应的“稠密等效模型 (Dense Equivalent)”参数量。以 DeepSeek-V3 架构为例：256 个路由专家 + 1 个共享专家，每次激活 8 个，`d_model=7168`。其总参数量令人咋舌，但活跃参数量仅为稠密版 Llama 3 70B 的七分之一。

## 使用指南

Hugging Face 加载方式：

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")

2026 年生产环境推理：vLLM 已原生支持 MoE 路由。SGLang 提供了最快的专家并行 (Expert Parallelism) 路径。两者均可自动处理 top-k 选择与专家并行。

**何时选择 MoE：**
- 希望在降低单 Token 推理成本的同时，获得前沿模型质量。
- 具备充足的显存 (VRAM) 或专家并行基础设施。
- 你的工作负载属于 Token 密集型（如对话、代码生成），而非上下文密集型（如长文档处理）。

**何时不应选择 MoE：**
- 边缘端部署——任何活跃浮点运算 (FLOP) 都需要承担完整的模型存储开销。
- 对延迟极度敏感的单用户服务——专家路由会引入额外开销。
- 小型模型（<7B）——MoE 的质量优势仅在算力超过一定阈值（约 60 亿活跃参数）后才会显现。

## 交付上线

参见 `outputs/skill-moe-configurator.md`。该技能模块会根据参数预算、训练 Token 数量以及部署目标，为新 MoE 模型自动选择专家数量 (E)、激活专家数 (k) 以及共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失（auxiliary-loss-free）的偏置更新如何在 50 次迭代中使专家（expert）的使用趋于均衡。
2. **中等。** 将可学习的路由器（learned router）替换为基于哈希的路由器（hash-based router）（确定性，无需学习）。对比生成质量与负载均衡效果。为什么可学习的路由器表现更好？
3. **困难。** 实现 GRPO 风格的“rollout 匹配路由”（rollout-matched routing）（DeepSeek-V3.2 技巧）：记录推理（inference）阶段激活的专家，并在梯度计算（gradient computation）阶段强制使用相同的路由。在简易的策略梯度（policy-gradient）实验环境中评估其效果。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 专家（Expert） | “众多前馈网络（FFN）中的一个” | 一个独立的前馈网络；其参数专门用于处理 FFN 计算中的稀疏切片。 |
| 路由器（Router） | “门控机制” | 一个小型线性层，用于对每个词元（token）与各个专家进行打分；执行 Top-k 选择（top-k selection）。 |
| Top-k 路由（Top-k routing） | “每个词元激活 k 个专家” | 每个词元的 FFN 计算恰好经过 k 个专家，权重由门控值决定。 |
| 辅助损失（Auxiliary loss） | “负载均衡惩罚项” | 额外的损失项，用于惩罚专家使用不均衡的情况。 |
| 无辅助损失（Auxiliary-loss-free） | “DeepSeek-V3 的技巧” | 仅通过路由器选择中的每个专家偏置来实现均衡；无需额外梯度。 |
| 共享专家（Shared expert） | “始终激活” | 每个词元都会经过的额外专家；用于捕获通用知识。 |
| 专家并行（Expert parallelism） | “按专家分片” | 将不同的专家分配到不同的 GPU 上；在网络中路由词元。 |
| 稀疏度（Sparsity） | “激活参数量 < 总参数量” | 比率 `k × expert_size / (E × expert_size)`；对于 DeepSeek-V3 约为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) —— 混合专家（Mixture-of-Experts）概念的奠基之作。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) —— Switch Transformer，经典的 MoE 架构。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) —— Mixtral 8×7B 模型。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) —— 包含 MLA、无辅助损失 MoE 及 MTP 技术。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) —— 基于偏置的负载均衡策略论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) —— 本课路由器所采用的细粒度划分与共享专家架构。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) —— 共享专家概念的原始论文。