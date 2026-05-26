# 键值缓存（KV Cache）、闪存注意力（Flash Attention）与推理优化（Inference Optimization）

> 训练是并行且受限于算力（FLOP-bound）的。推理是串行且受限于访存（memory-bound）的。瓶颈不同，优化策略自然也不同。

**类型：** 构建
**语言：** Python
**前置知识：** 第 7 阶段 · 02（自注意力机制），第 7 阶段 · 05（完整 Transformer），第 7 阶段 · 07（GPT）
**预计时间：** 约 75 分钟

## 问题所在

一个朴素的自回归解码器（autoregressive decoder）在生成 `N` 个词元（token）时需要执行 `O(N²)` 的计算量：在每一步中，它都会对整个前缀（prefix）重新计算注意力（attention）。对于一个 4K 词元的回复，这意味着 1600 万次注意力操作，其中绝大多数都是冗余的。前缀词元的每一个隐藏状态（hidden state）在计算完成后就是确定性的——你只需要将新词元的查询向量（query）与之前所有已缓存的键（keys）和值（values）进行计算即可。

除此之外，注意力机制本身会移动大量数据。标准的注意力机制会实例化一个 N×N 的分数矩阵、N×d 的 softmax 输出以及 N×d 的最终输出——这会导致对高带宽内存（HBM）进行过多的读写操作。当 N≥2K 时，注意力计算在成为算力瓶颈之前就会先成为访存瓶颈。传统的注意力算子（kernels）对现代 GPU 的利用率低了 4 到 10 倍。

两项优化技术（均出自 Dao 等人）将前沿模型的推理速度从“缓慢”推向了“快速”：

1. **键值缓存（KV Cache）。** 存储每个前缀词元的 K 和 V 向量。每个新词元的注意力计算只需将查询向量与已缓存的键进行一次匹配。这使得每一步生成的推理复杂度从 `O(N²)` 降至 `O(N)`。
2. **闪存注意力（Flash Attention）。** 对注意力计算进行分块（tile），使得完整的 N×N 矩阵永远不会写入 HBM。所有的 softmax 和矩阵乘法（matmul）都在静态随机存取存储器（SRAM）中完成。在 A100 上可实现 2–4 倍的实际运行加速（wall-clock speedup）；在 H100 上配合 FP8 精度可实现 5–10 倍的加速。

到 2026 年，这两项技术已成为行业标准。每一个生产级推理栈（inference stack）（如 vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认依赖它们。每一个前沿模型在发布时都已启用 Flash Attention。

## 核心概念

![KV 缓存增长与 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存 (KV Cache) 计算

每个解码器层 (Decoder Layer)、每个 Token、每个注意力头 (Attention Head)：

bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V

对于 7B 模型（32 层，32 个注意力头，d_head=128，fp16 精度）：

per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB

对于 Llama 3 70B（80 层，d_head=128，采用 8 个 KV 头的分组查询注意力 GQA (Grouped-Query Attention)）：

per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB

这 10 GB 的显存占用解释了为何在 128K 上下文长度下，即使批次大小 (Batch Size) 仅为 1，Llama 3 70B 也需要占用一块 40 GB A100 显卡的大部分显存来存放 KV 缓存。

**GQA 是 KV 缓存优化的关键。** 若使用 64 个头的多头注意力 MHA (Multi-Head Attention)，显存占用将高达 32 GB。而多潜在注意力 MLA (Multi-Latent Attention) 的压缩效果则更为显著。

### Flash Attention —— 分块 (Tiling) 技巧

标准注意力机制 (Standard Attention)：

S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)

需要三次高带宽内存 HBM (High Bandwidth Memory) 往返读写。在 H100 显卡上，HBM 带宽为 3 TB/s，而静态随机存取存储器 SRAM (Static Random-Access Memory) 带宽高达 30 TB/s。每次访问 HBM 相比将所有数据保留在芯片内，都会带来约 10 倍的性能损耗。

Flash Attention 实现：

for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM

每个分块仅需一次 HBM 访问。总内存占用从 `O(N²)` 降至 `O(N)`。反向传播 (Backward Pass) 通过重新计算前向传播 (Forward Pass) 的部分值来替代存储它们，这进一步节省了显存。

**数值计算技巧。** 运行中的 Softmax 会在各个分块间维护 `(max, sum)` 状态，从而确保最终的归一化结果精确无误。这并非近似计算——Flash Attention 的输出与标准注意力机制在比特级完全一致（仅受 fp16 浮点数非结合性影响）。

**版本演进：**

| 版本 | 年份 | 核心改进 | 参考硬件加速比 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 基于 SRAM 的分块内核 | A100 上提升 2 倍 |
| Flash 2 | 2023 | 更优的并行度，因果优先排序 | A100 上提升 3 倍 |
| Flash 3 | 2024 | Hopper 架构异步特性，FP8 支持 | H100 上提升 1.5–2 倍（FP16 约 740 TFLOPs） |
| Flash 4 | 2026 | Blackwell 五级流水线，软件 exp2 实现 | 优先面向推理（初期仅支持前向传播） |

Flash 4 发布初期仅支持前向传播。模型训练仍沿用 Flash 3。Flash 4 对 GQA 和变长序列 (Variable Length, varlen) 的支持仍在开发中（预计 2026 年中）。

### 投机解码 (Speculative Decoding) —— 降低延迟的另一利器

轻量级草稿模型 (Draft Model) 预生成 N 个 Token。大型验证模型并行验证这 N 个 Token。若验证通过 k 个 Token，则相当于仅消耗 1 次大模型前向传播的开销，就生成了 k 个 Token。在代码和散文生成任务中，典型的 k 值为 3–5。

2026 年主流方案：
- **EAGLE 2 / Medusa。** 集成草稿头，与验证模型共享隐藏状态。在无损质量的前提下实现 2–3 倍加速。
- **基于草稿模型的投机解码。** 在消费级硬件上可实现 2–4 倍加速。
- **前瞻解码 (Lookahead Decoding)。** 基于雅可比迭代 (Jacobi Iteration)，无需草稿模型。虽属小众方案，但完全免费。

### 连续批处理 (Continuous Batching)

传统批处理推理：需等待最慢的序列生成完毕，再启动新批次。当短回复提前结束时，会导致 GPU 算力闲置浪费。

连续批处理（最初随 Orca 发布，现已集成至 vLLM、TensorRT-LLM、SGLang 等框架）：旧请求一旦完成，立即将新请求插入批次中。在典型聊天工作负载下，吞吐量可提升 5–10 倍。

### PagedAttention —— 将 KV 缓存虚拟内存化

vLLM 的核心亮点功能。KV 缓存以 16 个 Token 为块进行分配，并通过页表 (Page Table) 将逻辑位置映射到物理块。该机制支持在并行采样（如束搜索 Beam Search、并行采样）间共享 KV 缓存，支持提示词缓存 (Prompt Caching) 的前缀热替换，并能有效消除内存碎片。相比传统的连续分配方式，吞吐量提升达 4 倍。

## 动手构建

参见 `code/main.py`。我们实现了：

1. 一个朴素的 `O(N²)` 增量解码器 (incremental decoder)。
2. 一个 `O(N)` 的 KV 缓存解码器 (KV-cached decoder)。
3. 一个分块 Softmax (tiled softmax)，用于模拟 Flash Attention 的运行最大值 (running-max) 算法。

### 步骤 1：KV 缓存 (KV cache)

class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]

原理很简单：在按层和按注意力头划分的列表中，持续追加每个 token 的 K、V 向量。

### 步骤 2：分块 Softmax (tiled softmax)

def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]

该实现能一次性输出与 `softmax(qK) V` 逐位完全一致的结果，但在任意时刻，其工作集 (working set) 仅为一个 `tile × d_head` 大小的分块，而非完整的 `N × d_head` 矩阵。

### 步骤 3：在生成 100 个 token 时对比朴素解码与缓存解码

统计注意力计算操作次数。朴素方法：`O(N²)` = 5050。缓存方法：`O(N)` = 100。代码会同时打印这两项结果。

## 实际应用

# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically

vLLM 生产环境部署：

pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8

跨请求的前缀缓存 (prefix caching) 是 2026 年的一项重大优势——相同的系统提示词 (system prompt)、少样本示例 (few-shot examples) 或长上下文文档可在多次调用间复用 KV 缓存。对于包含重复工具提示词的智能体 (agent) 工作负载，前缀缓存通常能带来 5 倍的吞吐量提升。

## 部署交付

参见 `outputs/skill-inference-optimizer.md`。该技能 (skill) 会为新的推理部署任务自动选择注意力实现方案、KV 缓存策略、量化方法以及投机解码 (speculative decoding) 配置。

## 练习

1. **简单。** 运行 `code/main.py`。验证基础解码器（naive decoder）与缓存解码器（cached decoder）的输出是否一致，并记录两者在操作次数（op-count）上的差异。
2. **中等。** 实现前缀缓存（prefix caching）：给定一个提示词（prompt）P 和多个续写结果（completions），先对 P 执行一次前向传播（forward pass）以填充 KV 缓存（KV cache），随后针对每个续写结果进行分支计算。对比该方案与为每个续写结果重新编码 P 的加速比（speedup）。
3. **困难。** 实现一个简易版分页注意力机制（PagedAttention）：使用固定 16 个 token 的块（blocks）与空闲链表（free-list）管理 KV 缓存。当序列（sequence）生成结束时，将其占用的块归还至内存池。模拟 1,000 次长度不一的对话生成任务，对比该方案与连续分配（contiguous allocation）在内存碎片化（memory fragmentation）方面的差异。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| KV 缓存 (KV cache) | “让解码变快的秘诀” | 存储每个前缀 token 的 K 和 V 值；新的查询（query）直接对这些缓存值进行注意力计算，而无需重新计算。 |
| 高带宽内存 (HBM) | “GPU 主内存” | High Bandwidth Memory；H100 为 80 GB，B200 为 192 GB。带宽约 3 TB/s。 |
| 静态随机存取存储器 (SRAM) | “片上内存” | 每个流多处理器（SM）的专用高速内存；H100 上每个 SM 约 256 KB。带宽约 30 TB/s。 |
| Flash Attention | “分块注意力内核” | 无需在 HBM 中显式构建 N×N 矩阵即可计算注意力机制。 |
| 连续批处理 (Continuous batching) | “无等待批处理” | 在不清空整个批次（batch）的情况下，动态替换已完成的序列并插入新序列。 |
| 分页注意力机制 (PagedAttention) | “vLLM 的核心亮点” | 通过页表（page table）以固定块分配 KV 缓存；彻底消除内存碎片。 |
| 前缀缓存 (Prefix caching) | “复用长提示词” | 跨请求缓存共享前缀的 KV 值；大幅降低智能体（agent）的运行成本。 |
| 投机解码 (Speculative decoding) | “草稿 + 验证” | 由轻量级草稿模型（draft model）预生成候选 token，再由大模型在一次前向传播中批量验证 k 个 token。 |

## 延伸阅读

- [Dao 等人 (2022)。FlashAttention：具备 IO 感知能力的快速且内存高效的精确注意力机制](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023)。FlashAttention-2：通过更优的并行性 (Parallelism) 与任务划分 (Work Partitioning) 实现更快的注意力机制](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah 等人 (2024)。FlashAttention-3：结合异步 (Asynchrony) 与低精度 (Low-precision) 实现的快速精准注意力机制](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 发布说明 (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 五级流水线 (5-stage pipeline) 与软件 exp2 优化技巧 (software-exp2 trick)；关于本课程提及的仅前向传播启动 (forward-only launch) 注意事项，请查阅该仓库的 README 文件。
- [Kwon 等人 (2023)。基于 PagedAttention 的大语言模型服务高效内存管理](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan 等人 (2023)。通过投机解码 (Speculative Decoding) 实现 Transformer 快速推理](https://arxiv.org/abs/2211.17192) — 投机解码 (spec decoding) 相关文献。
- [Li 等人 (2024)。EAGLE：投机采样需重新思考特征不确定性](https://arxiv.org/abs/2401.15077) — 本课程引用的集成草稿策略 (integrated-draft approach) 对应的 EAGLE-1/2 论文。
- [Cai 等人 (2024)。Medusa：基于多解码头 (Multiple Decoding Heads) 的简易大语言模型推理加速框架](https://arxiv.org/abs/2401.10774) — 与 EAGLE 一同被引用的 Medusa 方法。
- [vLLM 文档 — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 针对 16 词元块 (16-token block) 与页表设计 (page-table design) 的权威深度解析。