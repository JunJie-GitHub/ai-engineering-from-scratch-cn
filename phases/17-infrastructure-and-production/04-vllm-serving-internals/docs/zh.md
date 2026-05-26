# vLLM 服务内部机制：分页注意力 (PagedAttention)、连续批处理 (Continuous Batching) 与分块预填充 (Chunked Prefill)

> vLLM 在 2026 年的主导地位并非依赖单一技巧，而是建立在三个叠加的默认配置之上。分页注意力 (PagedAttention) 始终处于开启状态。连续批处理 (Continuous Batching) 会在解码迭代 (decode iteration) 之间将新请求注入活跃批次。分块预填充 (Chunked Prefill) 将长提示词切片，确保解码令牌 (decode token) 永远不会“饥饿”。同时开启这三项功能后，在单张 H100 SXM5 上运行 Llama 3.3 70B FP8 模型，在 128 并发下可推送 2,200-2,400 tok/s 的吞吐量——这比 vLLM 自身的默认配置高出约 25%，是朴素 PyTorch 循环的 3-4 倍。本教程将深入剖析调度器 (scheduler) 和注意力内核 (attention kernel)，达到你可以绘制架构图的程度，并在 `code/main.py` 中提供一个玩具级连续批处理调度器，演示 vLLM 如何调度预填充 (prefill) 和解码 (decode)。

**Type:** 学习
**Languages:** Python（标准库，玩具级连续批处理调度器）
**Prerequisites:** 第 17 阶段 · 01（模型服务），第 11 阶段（大语言模型工程）
**Time:** 约 75 分钟

## 学习目标

- 将分页注意力 (PagedAttention) 解释为一种键值缓存 (KV cache) 分配器：说明块 (blocks)、块表 (block tables) 的工作原理，以及为何在生产负载下内存碎片化 (fragmentation) 能保持在 4% 以下。
- 在迭代级别绘制连续批处理 (Continuous Batching) 的架构图：说明已完成的序列如何离开批次，新序列如何加入而无需清空批次。
- 用一句话描述分块预填充 (Chunked Prefill)，并指出它保护的是哪项延迟指标（提示：是首字延迟 (TTFT) 的尾部延迟，而非平均吞吐量）。
- 指出 2026 年 vLLM v0.18.0 中一个常见的陷阱，该陷阱会困扰那些试图一次性开启所有优化功能的团队。

## 问题背景

朴素的 PyTorch 服务循环一次只能处理一个请求：分词、预填充 (prefill)、解码 (decode) 直到遇到结束符 (EOS)，然后返回。面对单个用户时这没问题。但面对一百个用户时，它就变成了一个排长队的系统。显而易见的修复方案——静态批处理 (static batching)——会将窗口内每个请求填充到最长提示词的长度，将每次解码填充到预期最长输出的长度，并让整个批次等待最慢的序列。你为从未使用的填充付出了代价，而快速请求只能为慢速请求让路。

vLLM 同时解决了这三个问题。分页注意力 (PagedAttention) 阻止了键值缓存 (KV cache) 碎片化像传统连续分配那样吞噬 60-80% 的 GPU 显存。连续批处理 (Continuous Batching) 允许请求在每次解码迭代之间加入或离开批次，从而确保批次始终充满实际工作负载。分块预填充 (Chunked Prefill) 将 32k 令牌的提示词切分为约 512 令牌的片段，并与解码过程交错执行，从而避免长提示词冻结 GPU 上的每一个解码令牌。

2026 年的生产环境默认配置是同时开启这三项功能。你需要理解每一项的作用，因为它们的故障模式都出现在调度器 (scheduler) 层面，而非模型本身。

## 核心概念

### 分页注意力机制（PagedAttention）作为虚拟内存系统

每个序列的键值缓存（KV cache）大小为 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 8192 个 token 的 Llama 3.3 70B 模型，在 BF16 精度下，每个序列大约占用 1.25 GB。如果为每个请求预先保留 8192 个槽位，但平均请求仅使用 1500 个 token，你将浪费约 82% 预留的高带宽内存（HBM）。传统批处理（classic batching）方式会直接承担这种浪费。

分页注意力机制借鉴了操作系统虚拟内存的思想。每个序列的 KV cache 并非连续分配。它按固定大小的块（block）进行分配（默认为 16 个 token）。每个序列都维护一个块表（block table），用于将其逻辑 token 位置映射到物理块 ID。当序列长度超出已分配的块时，系统会追加一个新块。当序列处理完毕时，其占用的块将归还至内存池。

内存碎片率从传统方式的 60-80% 降至分页注意力机制的 4% 以下。你无需通过命令行标志来启用它——它是 vLLM 内置的唯一内存分配器。可调节的参数是 `--gpu-memory-utilization`（默认值为 0.9），它用于指示 vLLM 在加载模型权重和激活值后，预留多少 HBM 用于 KV 块。

### 迭代级别的连续批处理（Continuous batching）

旧的“动态批处理”会等待一个时间窗口（例如 10 毫秒）来凑满一个批次，然后执行预填充（prefill）+ 解码（decode）+ 解码 + 解码，直到所有序列处理完毕。处理较快的序列会提前结束并处于空闲状态，等待 GPU 完成较慢序列的计算。

连续批处理在每次解码步骤之间运行。我们将正在运行的序列集合称为 `RUNNING` 列表。在每次迭代中：

1. 移除 `RUNNING` 列表中刚刚遇到结束符（EOS）或达到最大 token 数（`max_tokens`）的序列。
2. 调度器（scheduler）检查等待队列。如果存在空闲的 KV 块，则接纳新序列（进行预填充或恢复运行）。
3. 对当前 `RUNNING` 列表中的所有序列执行前向传播（forward pass），每个序列生成一个新 token。

批次大小永远不会填充到固定数值。处于不同输出位置的序列共享一次融合前向计算（fused forward）。在 2026 年的 vLLM 版本中，这被称为 `V1 scheduler`。核心不变量：调度器在每个解码迭代运行一次，而非每个请求运行一次。

### 分块预填充（Chunked prefill）保护首词延迟（TTFT）尾部

预填充阶段受限于计算能力（compute-bound）。在单张 H100 上，Llama 3.3 70B 处理 32k token 的提示词需要约 800 毫秒的纯预填充时间。在预填充运行期间，批次中其他所有序列的解码 token 生成都会等待。在服务循环中，一个长提示词的首词延迟会成为其他数十个用户的词间延迟（ITL）尖峰。

分块预填充将预填充过程拆分为固定大小的块（默认为 512 个 token），并将每个块作为独立单元进行调度。在处理块与块之间的间隙，调度器可以推进解码序列生成一个 token。你以微小的绝对预填充延迟增加（每个块增加几毫秒）为代价，换取大幅降低的解码时间抖动。在已发布的基准测试中，混合负载下的 P99 ITL 从约 50 毫秒降至约 15 毫秒。

### 三大默认特性的协同作用

这三项特性彼此依赖、互为前提。分页注意力机制为调度器提供了细粒度的 KV 资源，以便进行灵活调度。连续批处理需要这种细粒度资源，从而在接纳新序列时无需触发全局重排。分块预填充是调度器在同一 `RUNNING` 列表上做出的决策——它只是一项额外的调度策略，而非独立的系统。

你无需了解每一个命令行参数。你需要理解调度器的优化目标：在 KV 块预算限制下，结合分块预填充的切片策略，最大化有效吞吐量（goodput）。

### 2026 年 v0.18.0 版本的注意事项

在 vLLM v0.18.0 中，无法将 `--enable-chunked-prefill` 与基于草稿模型的投机解码（speculative decoding，参数 `--speculative-model`）结合使用。文档中注明的唯一例外是 V1 调度器中的 N-gram GPU 投机解码。未阅读发布说明就盲目开启所有参数的团队，会在启动时直接遇到运行时错误（run-time error），而非性能轻微下降。如果投机解码带来的收益值得你开启分块预填充，请重新评估该选择——2026 年的正确做法通常是使用不带分块预填充的 EAGLE-3，而不是搭配无法编译的草稿模型与分块预填充。

### 关键性能数据

- Llama 3.3 70B FP8 精度，H100 SXM5，128 并发，三项特性全开：2,200-2,400 tok/s。
- 相同模型，默认 vLLM（未开启分块预填充）：约 1,800 tok/s。
- 相同模型，原生 PyTorch 前向循环：约 600 tok/s。
- 生产负载下，分页注意力机制的 KV 碎片浪费：<4%。
- 混合负载下的 P99 ITL：开启分块预填充约 15 毫秒，未开启约 50 毫秒。

### 调度器代码逻辑示例

while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call

`code/main.py` 正是使用标准库 Python 实现的上述循环，其中包含模拟的 token 数量与前向延迟。运行该脚本可以直观展示分块预填充如何在长预填充过程中保持解码序列的活跃状态。

## 使用它

`code/main.py` 模拟了一个具备可切换功能的 vLLM 风格调度器 (vLLM-style scheduler)。运行该脚本可查看：

- `NAIVE` 模式：每次仅处理一个请求，无批处理 (batching)。
- `STATIC` 模式：填充并等待 (pad and wait)，属于经典批处理。
- `CONTINUOUS` 模式：在每次迭代 (iteration) 级别进行请求的准入与释放。
- `CONTINUOUS + CHUNKED` 模式：将预填充 (prefill) 切片与解码 (decode) 过程交错执行。

输出结果将展示总吞吐量 (throughput)（以每秒虚拟词元数计）、平均首字延迟 (TTFT) 以及 P99 词元间延迟 (ITL)。在混合流量 (mixed traffic) 场景下，`CONTINUOUS + CHUNKED` 模式的性能应占据主导地位。

## 交付成果

本课时将生成 `outputs/skill-vllm-scheduler-reader.md` 文件。在给定服务配置 (serving config)（包括批处理大小、KV 内存利用率 (KV memory utilization)、分块预填充大小、推测解码配置 (speculative config)）的情况下，该脚本会输出一份调度器诊断报告 (scheduler diagnosis)，明确指出三个默认配置中哪一个构成了性能瓶颈，并给出相应的调优建议。

## 练习

1. 运行 `code/main.py`。在包含长短请求混合的工作负载 (workload) 上，对比 `STATIC` 与 `CONTINUOUS` 模式。吞吐量差距的来源是什么——是预填充效率、解码效率，还是尾部延迟 (tail latency)？
2. 修改示例调度器以添加 `--max-num-batched-tokens` 参数。对于运行 Llama 3.3 70B FP8 的 H100 显卡，该参数的合理取值是多少？（提示：该值取决于 KV 块大小和空闲块数量，而非原始 HBM 容量。）
3. 重新阅读 vLLM v0.18.0 的发布说明。哪些命令行标志 (flags) 的组合是互斥的？请列出它们。
4. 计算以下场景下的 KV 缓存碎片化浪费 (KV cache fragmentation waste)：针对包含 1,000 个请求的流量轨迹 (trace)，其输出词元数均值为 1,500，标准差为 600。请分别在 (a) 按请求连续分配 (contiguous per-request allocation) 且最大上限为 8192，以及 (b) 采用 16 词元块的 PagedAttention 机制下进行计算。
5. 用一段话解释为何分块预填充 (chunked prefill) 有助于降低 P99 ITL，但在独立运行时却无法提升吞吐量。在实际部署中，吞吐量的优势究竟从何而来？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| PagedAttention | “KV 缓存技巧” | KV 缓存的定长块分配器；碎片率 <4% |
| 块表 (Block table) | “页表” | 每个序列的逻辑词元位置到物理 KV 块的映射表 |
| 连续批处理 (Continuous batching) | “正确的动态批处理” | 在每次解码迭代时做出请求准入与释放的决策 |
| 分块预填充 (Chunked prefill) | “预填充拆分” | 将长预填充任务拆分为 512 词元的切片，并与解码过程交错执行 |
| 首字延迟 (TTFT) | “首个词元生成时间” | 预填充 + 排队 + 网络传输时间；在长提示词场景下主要由预填充阶段主导 |
| 词元间延迟 (ITL) | “词元间隔时间” | 连续解码词元之间的时间间隔；主要由批处理大小决定 |
| 有效吞吐量 (Goodput) | “满足服务等级目标的吞吐量” | 在确保每个请求均达到 TTFT 和 ITL 目标前提下的词元/秒速率 |
| V1 调度器 (V1 scheduler) | “新版调度器” | vLLM 的 2026 版调度器；N-gram 推测解码是与分块预填充兼容的路径 |
| `--gpu-memory-utilization` | “内存调节旋钮” | 在扣除模型权重和激活值后，预留给 KV 块的 HBM 显存比例 |

## 延伸阅读

- [vLLM 文档 — 推测解码（Speculative Decoding）](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于分块预填充（Chunked Prefill）与推测解码兼容性的官方资料。
- [vLLM 发布说明（NVIDIA）](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年发布周期（Release Cadence）及各版本特定行为说明。
- [vLLM 博客 — 分页注意力（PagedAttention）](https://blog.vllm.ai/2023/06/20/vllm.html) — 奠定当前内存分配器（Allocator）设计思路的原始技术文章。
- [PagedAttention 论文（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180) — 内存碎片（Fragmentation）分析与调度器（Scheduler）设计。
- [Aleksa Gordic — 深入 vLLM](https://www.aleksagordic.com/blog/vllm) — 结合火焰图（Flame Graphs）对 V1 调度器进行的详细解析。