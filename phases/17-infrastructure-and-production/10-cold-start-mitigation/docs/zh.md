# 无服务器大语言模型（Serverless LLM）的冷启动缓解（Cold-Start Mitigation）策略

> 一个 20 GB 的模型镜像从冷启动到提供服务，7B 模型需要 5-10 分钟，70B 模型则需要 20 分钟以上。在真正的无服务器（Serverless）架构中，这不算预热，而是服务中断。缓解策略在五个层级上运作：预置节点镜像（Pre-seeded Node Images，AWS 上的 Bottlerocket、双卷架构）、模型流式传输（Model Streaming，如 NVIDIA Run:ai Model Streamer 及 vLLM 原生支持）、GPU 内存快照（GPU Memory Snapshots，如 Modal 检查点，重启速度最高提升 10 倍）、预热池（Warm Pools，`min_workers=1`）、分层加载（Tiered Loading，如 ServerlessLLM 的 NVMe→DRAM→HBM 流水线，延迟降低 10-200 倍），以及实时迁移（Live Migration，仅迁移输入令牌（Input Tokens，KB 级）而非键值缓存（KV Cache，GB 级））。Modal 公布的冷启动时间底线为 2-4 秒；Baseten 默认为 5-10 秒，通过预热可实现亚秒级。本课程将教你如何测量、规划预算并组合运用这五个层级。

**类型：** 学习
**语言：** Python（标准库，玩具级冷启动路径模拟器）
**前置条件：** 第 17 阶段 · 02（推理平台经济学（Inference Platform Economics）），第 17 阶段 · 03（GPU 自动扩缩容（GPU Autoscaling））
**时长：** 约 60 分钟

## 学习目标

- 列举冷启动缓解的五个层级，并为每个层级命名一种工具或模式。
- 针对 70B 模型，将总冷启动时间计算为（节点配置）+（权重下载）+（权重加载至高带宽内存（HBM））+（引擎初始化）的总和。
- 解释为何实时迁移传输的是输入令牌（KB 级）而非键值缓存（GB 级），以及其代价是什么（重新计算）。
- 说明预热池的权衡取舍（为闲置 GPU 付费或接受冷启动长尾延迟），以及服务等级协议（SLA）阈值达到何种程度时 `min_workers > 0` 成为强制要求。

## 问题背景

你的无服务器大语言模型端点在夜间缩容至零。早上 8 点流量激增。第一个请求在以下过程中等待：

1. Karpenter 配置 GPU 节点：45-60 秒。
2. 容器拉取包含权重的 30 GB 镜像：120-300 秒。
3. 引擎将权重加载至高带宽内存：45-120 秒，具体取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA 图（CUDA Graphs）、键值缓存池（KV Cache Pool）和分词器（Tokenizer）：10-30 秒。

总计：220-510 秒（约 3-8 分钟）后才能返回第一个令牌（Token）。你的服务等级协议要求是 2 秒。你部署了一个预热池（`min_workers=1`），问题似乎消失了——但现在你需要为闲置的 GPU 全天候付费。如果你的服务有 5 个产品，每个产品各有一个预热副本，那么无论是否有用户调用，每月都将产生 5 × 24 × 30 = 3,600 个 GPU 小时的费用。

冷启动缓解的核心在于：如何在保持无服务器架构经济性的同时，尽可能接近常驻服务（Always-On）的低延迟表现。

## 核心概念

### 第 1 层 — 预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。在预拉取容器镜像后对数据卷进行快照，并在 `EC2NodeClass` 中引用该快照 ID。新节点启动时，模型权重已存在于本地 NVMe 存储中——镜像拉取（第 2 步）与部分权重加载（第 3 步）步骤随之消除。该方案与 Karpenter 原生兼容。典型收益：大型模型每次冷启动（Cold Start）可节省 2-4 分钟。

GCP 上的等效方案：使用预烘焙容器层的自定义虚拟机镜像。Azure 上的等效方案：采用相同模式的托管磁盘快照。

### 第 2 层 — 模型流式加载（Run:ai Model Streamer）

无需在响应首个请求前加载完整文件，而是将模型权重逐层流式传输至 GPU 显存，并在首个 Transformer 块驻留内存后立即开始处理。NVIDIA Run:ai Model Streamer 已作为原生组件集成于 vLLM 2026 版本中。支持 S3、GCS 和本地 NVMe。通过将 I/O 与计算初始化重叠执行，可将大型模型的权重加载时间缩短约一半。

### 第 3 层 — GPU 显存快照（Modal）

Modal 会在首次加载后对 GPU 状态（权重、CUDA 图、键值缓存区域（KV Cache））创建检查点（Checkpoint）。后续重启时可直接反序列化至高带宽内存（High Bandwidth Memory, HBM）——速度比重新初始化快 10 倍。这几乎等同于“2 秒内启动预热 GPU”。权衡点在于：快照与特定 GPU 拓扑（Topology）绑定，因此若 Karpenter 将工作负载迁移至不同规格的实例（SKU），则需重新创建检查点。

### 第 4 层 — 预热池（Warm Pool）（min_workers=1）

最直接的缓解方案：始终保持一个副本处于就绪状态。成本为单张 GPU 24x7 的每小时费率。成本核算对小型模型而言并不划算（为避免 30 秒冷启动需支付 0.85-1.50 美元/小时），但对大型模型则相对友好（为避免 5 分钟冷启动支付 4 美元/小时）。强制要求使用预热池的服务等级协议（Service Level Agreement, SLA）阈值通常为：70B 及以上模型的首词生成时间（Time To First Token, TTFT）P99 < 60 秒。

### 第 5 层 — 分层加载（Tiered Loading）（ServerlessLLM）

ServerlessLLM 将存储视为分层架构：NVMe（速度快但容量大）、动态随机存取存储器（Dynamic Random Access Memory, DRAM）（容量中等且分层）、HBM（容量小但访问极快）。权重预加载至 DRAM，按需加载至 HBM。相关论文指出，相较于直接将权重从磁盘加载至 HBM 的朴素方案，冷加载延迟可降低 10-200 倍。目前生产环境采用尚处早期，但已存在与 vLLM 的集成方案。

### 第 6 层 — 实时迁移（Live Migration）（附加模式）

当节点不可用时（如竞价实例回收（Spot Eviction）、节点排空（Node Drain）），传统做法是冷启动另一个副本并排空请求队列。实时迁移则将输入 Token（仅数千字节）传输至已加载模型的目标节点，并在目标节点上重新计算 KV Cache。重新计算的开销远低于通过网络传输 GB 级别的 KV Cache。该方案适用于解耦部署（Disaggregated Deployment）架构。

### 预热池的成本核算

对于 P99 TTFT SLA 要求为 2 秒的服务，核心问题不再是“是否使用预热池”，而是“需要多少预热副本，以及哪些请求路径应分配它们”。

- 高价值交互路径（实时聊天、语音智能体）：`min_workers=1-2`。
- 后台批处理路径（夜间分类任务）：可接受缩容至零（Scale-to-Zero），容忍 5-10 分钟的冷启动。
- 高级套餐：为每个租户配置专属容量的 `min_workers`。

### 优化前先进行度量

全新节点上 70B 模型的冷启动耗时拆解（示例数据）：

| 阶段 | 耗时 | 缓解方案 |
|-------|------|-----------|
| 节点配置 | 50s | Bottlerocket + 预置镜像、预热池 |
| 镜像拉取 | 180s | 预置数据卷（消除该步骤） |
| 权重加载至 HBM | 75s | 模型流式加载（减半）；GPU 快照（消除该步骤） |
| 引擎初始化 | 20s | 持久化 CUDA 图缓存 |
| 首次前向传播（Forward Pass） | 3s | 最低固有延迟 |
| **总冷启动耗时** | **328s** | |
| **采用缓解方案后总耗时** | **~15s** | 降低 22 倍 |

### 关键数据备忘

- Modal 冷启动：2-4 秒（配合 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预预热后可达亚秒级。
- 原始 70B 模型冷启动：3-8 分钟。
- Run:ai Model Streamer：权重加载速度提升约 2 倍。
- ServerlessLLM 分层加载：延迟降低 10-200 倍（论文数据）。

## 使用它

`code/main.py` 模拟了采用与不采用各项缓解措施（mitigation）时的冷启动（cold-start）路径。它会报告总冷启动时间、热池（warm pool）成本，以及热池能够收回自身成本的盈亏平衡请求率（break-even request rate）。

## 部署上线

本练习将生成 `outputs/skill-cold-start-planner.md`。该文件会根据服务等级协议（SLA）、模型规模及流量特征（traffic shape），推荐应叠加使用的缓解措施组合。

## 练习

1. 运行 `code/main.py`。计算盈亏平衡请求率，当请求率超过该值时，维持热副本（warm replica）的成本将低于因冷启动惩罚（cold-start tax）导致服务等级目标（SLO）违约而产生的额外请求丢弃成本。
2. 你部署了一个 13B 模型，要求 P99 首词元时间（TTFT）的 SLA 为 3 秒。请选择能满足该要求的最小缓解措施栈（层数最少）。
3. Bottlerocket 预置（pre-seeding）消除了容器镜像拉取步骤，但模型权重仍需从快照加载至高带宽内存（HBM）。若基于快照的 NVMe 读取速度为 7 GB/s，请计算 70B 模型的实际加载耗时（wall-clock time）。
4. 你的无服务器（serverless）提供商提供 GPU 快照（GPU snapshot）功能（例如 Modal），但你的团队以“快照会泄露个人身份信息（PII）”为由拒绝使用。请从正反两方面进行论述：实际风险究竟有多大？有哪些可行的缓解措施（如临时快照、加密、命名空间隔离）？
5. 设计一套分层热池（tiered warm-pool）策略：针对付费用户、试用用户和批处理工作负载（batch workload），分别应保留多少热副本？请列出具体的计算过程。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 冷启动（Cold start） | “漫长的停顿” | 从发起请求到全新副本生成首个词元（token）所需的时间 |
| 热池（Warm pool） | “始终在线的最低配置” | 设置 `min_workers >= 1` 以保持至少一个副本处于就绪状态 |
| 预置镜像（Pre-seeded image） | “预烘焙的 AMI” | 节点镜像中已预先驻留模型权重 |
| Bottlerocket | “AWS 节点操作系统” | AWS 专为容器优化的操作系统，支持双卷快照 |
| 模型流式加载器（Model streamer） | “流式加载” | 将权重 I/O 与计算环境初始化重叠执行 |
| GPU 快照（GPU snapshot） | “检查点存入 HBM” | 序列化加载后的 GPU 状态；重启时反序列化恢复 |
| 分层加载（Tiered loading） | “NVMe + DRAM + HBM” | 多级存储层次结构；按需加载 |
| 实时迁移（Live migration） | “转移词元” | 仅传输输入数据（KB 级），在目标节点重新计算键值（KV）缓存 |
| `min_workers` | “热副本” | 无服务器架构中的最低保活副本数 |
| 缩容至零（Scale-to-zero） | “完全无服务器化” | 空闲时无成本；但需承担完整的冷启动惩罚 |

## 延伸阅读

- [Modal — 冷启动性能 (Cold Start Performance)](https://modal.com/docs/guide/cold-start) — Modal 官方发布的基准测试 (Benchmarks) 与检查点架构 (Checkpoint Architecture)。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预置数据卷快照模式 (Pre-seeded Data Volume Snapshot Pattern)。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 权重加载与计算初始化重叠执行 (Overlap Weights Load with Compute Setup)。
- [Baseten — 冷启动缓解 (Cold-start Mitigation)](https://www.baseten.co/blog/cold-start-mitigation/) — 预热操作指南 (Pre-warming Playbook)。
- [ServerlessLLM 论文 (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计 (Tiered Loading Design)。
- [NVIDIA — 基于 Kubernetes 的解耦大语言模型推理 (Disaggregated LLM Inference on Kubernetes)](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 面向解耦部署的热迁移 (Live Migration for Disaggregated Deployments)。