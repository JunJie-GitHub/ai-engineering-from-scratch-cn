# 分离式预填充/解码 (Disaggregated Prefill/Decode) — NVIDIA Dynamo 与 llm-d

> 预填充 (Prefill) 属于计算密集型 (compute-bound) 任务，而解码 (Decode) 属于显存密集型 (memory-bound) 任务。将两者运行在同一张 GPU 上会导致其中一种资源被浪费。分离式架构 (Disaggregation) 将它们拆分到独立的资源池中，并通过 NIXL（支持 RDMA/InfiniBand 或 TCP 回退）在池间传输 KV 缓存 (KV cache)。NVIDIA Dynamo（GTC 2025 发布，1.0 GA 版本）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler 与 SLA Planner 可自动匹配预填充与解码的比例，以满足服务等级目标 (SLO)。NVIDIA 公布的吞吐量提升数据大致在此范围内：developer.nvidia.com (2025-06) 显示，在中等延迟场景下，GB200 NVL72 + Dynamo 运行 DeepSeek-R1 混合专家模型 (MoE) 的吞吐量提升约 6 倍；Dynamo 产品页 (developer.nvidia.com, undated) 宣称，GB300 NVL72 + Dynamo 对比 Hopper 架构，MoE 吞吐量最高可达 50 倍。“30 倍”这一数据是社区基于全栈 Blackwell + Dynamo + DeepSeek-R1 报告的汇总值；我们尚未找到明确标注恰好 30 倍的单一原始出处，因此请将其视为趋势性参考。llm-d（Red Hat + AWS）采用 Kubernetes 原生 (Kubernetes-native) 设计：将预填充/解码/路由器作为独立服务，并配置基于角色的水平 Pod 自动扩缩容 (HPA)。llm-d 0.5 版本新增了分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络以及缩容至零 (scale-to-zero) 功能。经济性分析：基于多项客户披露数据的内部汇总表明，在保持相同服务等级协议 (SLA) 的前提下，从集中式部署转向采用 Dynamo 的分离式架构，可为 200 万美元级别的推理支出节省 30%~40%（即每年节省 60~80 万美元）；该具体金额（200 万→60~80 万）为内部综合估算值，并非单一已发布的案例研究——请将其作为数量级参考，而非引用依据。短提示词（<512 tokens，短输出）场景无法抵消 KV 缓存的传输成本。

**Type:** 学习
**Languages:** Python（标准库，简易分离式与集中式对比模拟器）
**Prerequisites:** 第 17 阶段 · 04（vLLM 服务内部原理），第 17 阶段 · 08（推理指标）
**Time:** 约 75 分钟

## 学习目标

- 解释为何预填充与解码需要不同的最优 GPU 资源分配，并量化集中部署 (colocation) 下的资源浪费。
- 绘制分离式架构图：预填充池、解码池、通过 NIXL 传输 KV 缓存、路由器。
- 指出分离式架构不具成本效益的场景（短提示词、短输出）。
- 区分 NVIDIA Dynamo（上层堆栈架构）与 llm-d（Kubernetes 原生架构），并将其匹配至相应的运维场景。

## 问题背景

在 8 张 H100 GPU 上运行 Llama 3.3 70B 模型。在混合负载（长提示词 + 短输出）下，由于大部分计算资源已消耗在预填充阶段，GPU 在解码阶段会处于空闲状态。而在另一种负载（短提示词 + 长输出）下，情况则完全相反。将预填充与解码集中部署在同一节点，意味着你必须对两者都进行过度配置。

预算影响：20%~40% 的 GPU 运行时间被浪费在了错误的资源上。你实际上是在用 H100 的计算能力去运行显存密集型的解码任务，或者用 H100 的 HBM 带宽去运行计算密集型的预填充任务。这两种情况都是高昂的资源浪费。

分离式架构将预填充和解码拆分到独立的资源池中，并根据各自的瓶颈进行容量规划。KV 缓存通过高带宽互连网络从预填充池传输至解码池。

## 概念

### 瓶颈差异的原因

**Prefill（预填充）**——在一次前向传播中处理完整的输入提示词。矩阵乘法占主导地位；属于计算密集型（compute-bound）。H100 的 FP8 精度可提供约 2000 TFLOPS 的有效吞吐量。批处理效率较高——一次前向传播可处理大量 token。

**Decode（解码）**——每次生成一个 token，每次迭代都需读取完整权重。属于内存带宽密集型（memory-bandwidth-bound）。HBM3 提供约 3 TB/s 的带宽。仅在高并发时批处理效率才较好——权重读取的开销可分摊到整个批次。

将两者部署在同一节点（Colocating）：你需要购买对两者都优化的 GPU。H100 在两方面表现都不错，但成本固定。在大规模部署时，你通常希望将预填充池（prefill pool）部署在 H100 上（计算密集型）；将解码池（decode pool）部署在 H200 上（内存密集型），或采用激进的量化（quantization）策略。

### 架构设计

            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client

NIXL 是 NVIDIA 的节点间传输协议（inter-node transport）。在可用时使用 RDMA/InfiniBand，否则回退到 TCP。传输延迟是真实存在的——对于 70B FP8 模型，4K token 提示词的 KV 缓存（KV cache）传输通常需要 20-80 毫秒。这就是短提示词不适合采用分离架构（disaggregation）的原因：传输开销（transfer tax）会超过节省的成本。

### Dynamo 与 llm-d 对比

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA 版本）：
- 作为编排器（orchestrator）运行在 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 负责测量工作负载，SLA Planner 自动配置预填充与解码的比例（prefill:decode ratios）。
- 核心采用 Rust 编写，支持 Python 扩展。
- 吞吐量提升：NVIDIA 报告称，在中等延迟区间（medium-latency regime）内，GB200 NVL72 + Dynamo 运行 DeepSeek-R1 MoE 的吞吐量提升约 6 倍（developer.nvidia.com, 2025-06）；社区关于完整 Blackwell + Dynamo + DeepSeek-R1 栈“最高提升 30 倍”的说法缺乏单一权威来源，应视为趋势性参考（directional）。
- GB300 NVL72 + Dynamo：根据 Dynamo 产品页面数据，相比 Hopper 架构，MoE 吞吐量最高可达 50 倍（developer.nvidia.com, 未注明日期）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 将预填充、解码和路由器作为独立的 Kubernetes Service 部署。
- 基于角色的 HPA（Horizontal Pod Autoscaler，水平 Pod 自动扩缩容），使用队列深度（预填充）/ KV 利用率（解码）作为扩缩容信号。
- `topologyConstraint packDomain: rack` 将预填充与解码节点组（cliques）打包到同一机架，以实现高带宽的 KV 传输。
- llm-d 0.5（2026 年）：支持分层 KV 卸载（hierarchical KV offloading）、缓存感知的 LoRA 路由、UCCL 网络以及缩容至零（scale-to-zero）。

如果你希望使用托管式的上层编排栈（stack-above orchestrator），请选择 Dynamo。如果你需要 Kubernetes 原生原语（primitives）并致力于融入 CNCF 生态，请选择 llm-d。

### 经济效益

内部综合数据（非单一公开案例研究——量级参考基准）：
- 集中式部署（colocated serving）的推理年支出为 200 万美元。
- 切换至基于 Dynamo 的分离架构。
- 请求量不变，P99 延迟服务等级协议（SLA）保持不变。
- 报告节省成本：每年 60 万至 80 万美元（降低 30%–40%）。
- 无需新增硬件。

该数据综合自多位客户的披露信息，而非单一可引用的案例研究；最接近的公开数据点是 Baseten 使用 Dynamo KV 路由实现的 TTFT（Time To First Token，首字延迟）加快 2 倍 / 吞吐量提升 61%（baseten.co, 2025-10），以及 VAST + CoreWeave 预测在 40%–60% KV 命中率（KV hit rate）下，每美元 token 产出增加 60%–130%（vastdata.com, 2025-12）。节省的成本源于对每个资源池的精准容量规划（right-sizing）；预填充密集型工作负载（如带有 8K+ 前缀的 RAG 应用）比均衡型工作负载受益更大。

### 何时不应采用分离架构

- 提示词 < 512 token 且输出 < 200 token：传输开销将抵消性能收益。
- 小型集群（< 4 张 GPU）：资源池缺乏足够的多样性。
- 团队无法运维两个支持按角色独立扩缩容的 GPU 资源池：Dynamo 能提供帮助，但并非开箱即用。
- 缺乏 RDMA 网络架构：TCP 传输的开销会更重。

### 路由器与 Phase 17 · 11 的集成

分离式路由器具备 KV 缓存感知能力（Phase 17 · 11）。请求会被路由到持有其前缀的解码池——若无匹配，则按预填充 → 解码流程处理。命中率与分离架构的效果会相互叠加——缓存感知路由器决定了是否真的需要执行新的预填充。

### Blackwell 架构上的 MoE 才是真正体现数据优势的地方

GB300 NVL72 + Dynamo 相比 Hopper 基线展现出 50 倍的 MoE（Mixture of Experts，混合专家模型）吞吐量。MoE 的专家路由在预填充阶段属于计算密集型，而在解码阶段（依赖专家缓存）属于内存密集型，因此分离架构能带来双重收益。2026 年的前沿模型服务将以 MoE 为主导（如 DeepSeek-V3 及未来的 GPT-5 变体）。

### 需要牢记的关键数据

基准测试数据会随时间变化——NVIDIA 和推理栈每季度都会发布更新结果。引用前请务必重新核实。

- GB200 NVL72 + Dynamo 运行 DeepSeek-R1：在中等延迟区间内，吞吐量相比基线提升约 6 倍（developer.nvidia.com, 2025-06）；社区关于完整 Blackwell + Dynamo 栈“最高提升 30 倍”的说法属于趋势性汇总，缺乏单一权威来源。
- GB300 NVL72 + Dynamo：相比 Hopper 架构，MoE 吞吐量最高可达 50 倍（developer.nvidia.com, 未注明日期）。
- 节省成本基准（内部综合数据，非单一案例研究）：在 SLA 不变的前提下，每年 200 万美元的支出可节省 60 万至 80 万美元。
- 分离架构适用阈值：提示词 > 512 token 且输出 > 200 token。
- 通过 NIXL 进行 KV 传输：70B FP8 模型处理 4K 提示词的 KV 传输耗时为 20-80 毫秒。

## 使用指南

`code/main.py` 模拟了集中式推理服务（colocated serving）与分离式推理服务（disaggregated serving）的对比。该脚本会报告吞吐量（throughput）、单次请求成本（cost per request）以及提示词长度的性能交叉点（prompt-length crossover）。

## 交付上线

本章节将生成 `outputs/skill-disaggregation-decider.md` 文件。该文件会根据给定的工作负载（workload）和集群（cluster）配置，自动决策是否采用分离式架构。

## 练习

1. 运行 `code/main.py`。在什么提示词长度下，分离式部署的性能会优于集中式部署？
2. 为一个 P99 前缀长度（P99 prefix length）为 8K、输出长度为 300 的检索增强生成（RAG）服务设计预填充池（prefill pool）和解码池（decode pool）。
3. Dynamo 与 llm-d 对比：对于纯 Kubernetes 环境且无特定 Python 运行时偏好的团队，应选择哪一个？
4. 计算 KV 缓存传输（KV transfer）成本：在 70B FP8 模型上进行 4K 预填充（prefill）会产生约 500 MB 的 KV 缓存。在 RDMA 100 GB/s 带宽下，传输耗时为 5 ms；在 TCP 10 GB/s 带宽下，传输耗时为 50 ms。哪种延迟对你的服务等级协议（SLA）影响更大？
5. 混合专家模型（MoE）的专家路由（expert routing）会改变 KV 缓存的访问模式。当 MoE 架构针对每个词元（token）激活不同专家时，分离式部署的表现会如何？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 分离式推理服务（Disaggregated serving） | “拆分预填充/解码阶段” | 为每个阶段分配独立的 GPU 资源池 |
| NIXL | “NVIDIA 传输层” | Dynamo 的节点间 KV 缓存传输协议（支持 RDMA/TCP） |
| NVIDIA Dynamo | “编排器” | 位于 vLLM/SGLang/TRT-LLM 之上的堆栈协调器 |
| llm-d | “Kubernetes 原生” | Red Hat 与 AWS 联合推出的 K8s 分离式推理技术栈 |
| Planner Profiler | “Dynamo 自动配置” | 测量工作负载并配置资源池比例 |
| SLA Planner | “Dynamo 策略” | 自动匹配预填充与解码的速率比例，以满足服务等级目标（SLO） |
| `packDomain: rack` | “llm-d 拓扑配置” | 将预填充与解码节点部署在同一机架内，以实现高速 KV 传输 |
| UCCL | “统一集合通信” | llm-d 0.5 版本中用于支持缩容至零（scale-to-zero）的网络层 |
| MoE 专家路由（MoE expert routing） | “按词元分配专家” | DeepSeek-V3 采用的模式；分离式架构可有效优化此场景 |

## 延伸阅读

- [NVIDIA — 介绍 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — 在 Kubernetes 上部署分离式大语言模型推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM 分离式推理服务技术博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub 仓库](https://github.com/llm-d/llm-d)
- [llm-d 0.5 版本发布说明](https://github.com/llm-d/llm-d/releases)