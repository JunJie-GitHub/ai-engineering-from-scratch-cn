# Kubernetes 上的 GPU 自动扩缩容（GPU Autoscaling）— Karpenter、KAI Scheduler 与成组调度（Gang Scheduling）

> 三层架构，而非单一方案。Karpenter 动态执行节点供应（Node Provisioning）（耗时不到一分钟，比集群自动扩缩容器（Cluster Autoscaler）快 40%）。KAI Scheduler 负责成组调度（Gang Scheduling）、拓扑感知（Topology Awareness）与分层队列（Hierarchical Queues）——它能有效避免“8缺1”的部分分配陷阱（Partial Allocation Trap），即 7 个节点因等待 1 个缺失的 GPU 而空转烧钱。应用层自动扩缩容器（Application-level Autoscalers）（如 NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理专属指标进行扩缩容——例如队列深度（Queue Depth）与 KV 缓存利用率（KV Cache Utilization），而非 CPU/DCGM 的工作周期（Duty Cycle）。经典的 HPA（Horizontal Pod Autoscaler）陷阱在于，`DCGM_FI_DEV_GPU_UTIL` 仅衡量工作周期：100% 的利用率可能对应 10 个请求，也可能对应 100 个。vLLM 会预分配 KV 缓存内存，因此内存指标永远不会触发缩容（Scale-down）。本课程将教你如何将这三层架构组合使用，并避开 Karpenter 默认的 `WhenEmptyOrUnderutilized` 整合策略（Consolidation Policy），该策略会在推理中途终止正在运行的 GPU 任务。

**Type:** 学习
**Languages:** Python（标准库，简易队列深度自动扩缩容模拟器）
**Prerequisites:** 第 17 阶段 · 02（推理平台经济学），第 17 阶段 · 04（vLLM 服务内部原理）
**Time:** 约 75 分钟

## 学习目标

- 绘制三层自动扩缩容架构（节点供应、成组调度、应用层）的示意图，并指出每一层所使用的工具。
- 解释为何 `DCGM_FI_DEV_GPU_UTIL` 不适合作为 vLLM 的 HPA 信号，并列举两种替代指标（队列深度、KV 缓存利用率）。
- 描述成组调度（Gang Scheduling）的概念，以及 KAI Scheduler 所避免的部分分配故障模式（8 张 GPU 中 7 张闲置）。
- 指出会终止运行中 GPU 任务的 Karpenter 整合策略（`WhenEmptyOrUnderutilized`），并说明 2026 年的安全替代方案。

## 问题背景

你的团队在 Kubernetes 上部署了大语言模型（LLM）推理服务。你配置了 HPA，并将 `DCGM_FI_DEV_GPU_UTIL` 作为扩缩容信号。在工作时段，该服务的 GPU 利用率始终卡在 100%。HPA 却从未触发扩容——因为它认为资源已经满载。你手动添加了一个副本，首字延迟（Time To First Token, TTFT）随之降低，但 HPA 依然没有动作。这个指标在“欺骗”你。

另一方面，你使用集群自动扩缩容器（Cluster Autoscaler）来管理节点。凌晨 2 点，一个包含 100 万 token 的提示词（Prompt）请求到达；集群花费了 3 分钟来供应新节点，导致该请求超时。

此外，你部署了一个需要跨 2 个节点占用 8 张 GPU 的 70B 参数模型。集群中目前有 7 张 GPU 空闲，另有 1 张分散在 3 个节点上。集群自动扩缩容器为这缺失的 1 张 GPU 供应了一个新节点。在 Kubernetes 拉起最后一张 GPU 的 4 分钟里，7 个节点只能干等并持续烧钱。

三层架构，三种不同的故障模式。2026 年具备 GPU 感知（GPU-aware）能力的自动扩缩容绝不是“开启 HPA 就完事”，而是需要将节点供应、成组调度与应用信号扩缩容有机组合。

## 核心概念

### 第 1 层 — 节点供应 (Node Provisioning) (Karpenter)

Karpenter 会监控处于 Pending 状态的 Pod，并在约 45-60 秒内完成节点供应（集群自动扩缩容器 (Cluster Autoscaler) 针对 GPU 节点通常需要 90-120 秒）。它会根据 `NodePool` 约束动态选择实例类型——如果你的 Pod 需要 8 张 H100 GPU，而集群中没有匹配的节点，Karpenter 会直接供应一台新节点，而不是扩缩容现有的节点组。

**整合陷阱 (Consolidation Trap)**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 策略对 GPU 资源池而言非常危险。它会终止正在运行的 GPU 节点，以便将 Pod 迁移到更便宜且规格更匹配的实例上。对于推理工作负载，这意味着会驱逐正在处理的请求，并在新节点上重新加载一个 70B 参数规模的模型。其代价是数分钟的算力损失以及请求失败。

GPU 资源池的安全配置如下：

disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h

该配置允许 Karpenter 在一小时后整合真正空闲的节点，但绝不会驱逐正在运行的任务。

### 第 2 层 — 协同调度 (Gang Scheduling) (KAI Scheduler)

KAI Scheduler（原项目名为 "Karp"，后更名）负责处理默认 `kube-scheduler` 无法胜任的任务：

**协同调度 (Gang Scheduling)** —— 采用“全有或全无”的调度策略。一个需要 8 张 GPU 的分布式推理 Pod，要么 8 张同时启动，要么全部不启动。如果没有此机制，就会陷入“部分分配陷阱 (Partial-Allocation Trap)”：8 个 Pod 中有 7 个启动了，然后无限期等待，白白烧钱。

**拓扑感知 (Topology Awareness)** —— 识别哪些 GPU 共享 NVLink、哪些位于同一机架、哪些之间通过 InfiniBand 互联。并据此进行 Pod 放置。例如，DeepSeek-V3 67B 的张量并行 (Tensor Parallel) 工作负载必须保持在同一个 NVLink 域内；KAI Scheduler 会严格遵守这一要求。

**分层队列 (Hierarchical Queues)** —— 多个团队基于优先级和配额竞争同一个 GPU 资源池。团队 A 的生产环境紧急任务只有在优先级规则允许的情况下，才会被团队 B 的训练任务抢占。

KAI 作为二级调度器与 `kube-scheduler` 并行部署；你只需为工作负载添加注解 (Annotation) 即可使用它。Ray 和 vLLM 生产栈均已集成该调度器。

### 第 3 层 — 应用层信号 (Application-Level Signals)

**HPA 陷阱 (HPA Trap)**：`DCGM_FI_DEV_GPU_UTIL` 是一个占空比 (Duty-Cycle) 指标——它仅测量 GPU 在每个采样间隔内是否处于工作状态。100% 的利用率可能意味着 10 个并发请求，也可能意味着 100 个；GPU 反正都是忙碌的。基于占空比进行扩缩容等同于盲目扩缩容。

更糟糕的是，vLLM 及类似引擎会预分配 KV 缓存 (KV Cache) 内存（最高可达 `--gpu-memory-utilization` 设定的比例）。即使只有一个请求，内存使用率也会维持在 90% 左右。基于内存使用率的 HPA 将永远无法触发缩容。

**2026 年替代信号指标**：

- 队列深度 (Queue Depth)（等待预填充 (Prefill) 的请求数量）。
- KV 缓存利用率 (KV Cache Utilization)（已分配给活跃序列的内存块比例）。
- 单副本 P99 首字延迟 (TTFT, Time To First Token)（你的服务等级协议 (SLA) 信号）。
- 有效吞吐量 (Goodput)（每秒满足所有服务等级目标 (SLO) 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 会消费这些信号并据此扩缩容副本。在大语言模型 (LLM) 服务场景中，它们已完全取代 HPA。

### 工具选型指南

| 扩缩容决策 | 推荐工具 |
|----------------|------|
| 增加/移除节点 | Karpenter |
| 调度多 GPU 任务 | KAI Scheduler |
| 增加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级任务 | KAI Scheduler 队列 |

### 预填充与解码分离 (Disaggregated Prefill/Decode) 使一切变得复杂

如果你采用预填充与解码分离架构（第 17 阶段 · 17），你将拥有两类触发扩缩容条件不同的 Pod：预填充 Pod 根据队列深度进行扩缩容，解码 Pod 则根据 KV 缓存压力进行扩缩容。llm-d 将它们暴露为独立的 `Services`，并为每个角色配置独立的 HPA。切勿尝试在两者前面只部署一个统一的 HPA。

### 冷启动 (Cold Start) 问题在此同样关键

冷启动缓解策略（第 17 阶段 · 10）正是节点供应时间转化为用户可感知延迟的环节。Karpenter 45-60 秒的节点预热，加上加载 20GB 模型以及引擎初始化，意味着从零开始的请求将耗时 2-5 分钟。对于对 SLO 要求严格的路径，请保留一个热节点池（`min_workers=1`），或在应用层采用类似 Modal 的检查点 (Checkpointing) 技术。

### 关键数据备忘

- Karpenter 节点供应：约 45-60 秒，而 Cluster Autoscaler 针对 GPU 节点需 90-120 秒。
- KAI Scheduler 可避免部分分配造成的资源浪费——即“7/8 陷阱”。
- 将 `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：已失效；请改用队列深度或 KV 缓存利用率。
- Karpenter 的 `WhenEmptyOrUnderutilized` 策略：会终止正在运行的 GPU 任务。推理场景请使用 `WhenEmpty + consolidateAfter: 1h`。

## 使用它

`code/main.py` 模拟了在突发型 GPU 工作负载（bursty GPU workload）下的三层自动扩缩容器（autoscaler）机制。对比了基于占空比（duty cycle）的简单水平 Pod 自动扩缩容（HPA）、基于队列深度（queue-depth）的 HPA，以及采用 KAI 协同调度（KAI gang scheduling）的扩缩容策略。报告未满足的请求数、GPU 空闲分钟数以及综合评分。

## 交付它

本课时将生成 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑（cluster topology）、工作负载形态（workload shape）和服务等级目标（SLO），该脚本将设计一套三层自动扩缩容（autoscaling）方案。

## 练习

1. 运行 `code/main.py`。在突发型工作负载下，基于占空比的简单 HPA 会丢弃多少请求，而基于队列深度的 HPA 却能成功处理？这种差异源于何处？
2. 为在 H100 SXM5 上运行 Llama 3.3 70B FP8 的集群设计一个 Karpenter 节点池（NodePool）。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个用于防止非 GPU 工作负载调度到这些节点上的污点（taint）。
3. 你的团队报告部署卡在 Pending 状态，提示“GPU 可用但 Pod 无法调度”。请进行诊断——这是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标可以证实？
4. 分别为解耦的预填充（prefill）Pod 和解码（decode）Pod 选择自动扩缩容信号，并说明理由。
5. 计算 `WhenEmptyOrUnderutilized` 节点合并（consolidation）陷阱对一个 7x24 小时生产服务造成的成本影响。该服务平均每天发生 60 次请求丢弃事件，且 P99 首字延迟（TTFT）> 10 秒。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Karpenter | “节点供应器” | Kubernetes 节点自动扩缩容器（node autoscaler）；支持亚分钟级供应 |
| Cluster Autoscaler | “旧版扩缩容器” | Kubernetes 节点自动扩缩容器的前身；速度较慢，基于节点组（group-based） |
| KAI Scheduler | "GPU 调度器” | 用于协同调度（gang scheduling）+ 拓扑感知（topology awareness）+ 队列管理的二级调度器 |
| Gang scheduling | “全有或全无” | 原子性地调度 N 个 Pod，否则全部推迟 |
| Topology awareness | “机架感知” | 根据 NVLink/InfiniBand（IB）/机架位置来放置 Pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率” | 占空比（duty-cycle）指标；不适合作为大语言模型（LLM）的扩缩容信号 |
| Queue depth | “等待请求数” | 面向预填充（prefill）阶段扩缩容的正确 HPA 信号 |
| KV cache utilization | “内存压力” | 面向解码（decode）阶段扩缩容的正确 HPA 信号 |
| Consolidation | "Karpenter 合并” | 终止节点并替换为更便宜的实例类型 |
| `WhenEmpty + 1h` | “安全合并” | 不会驱逐正在运行的 GPU 任务的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档与配置示例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — 整合策略（Consolidation Policy）语义与 GPU 安全默认配置。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩容信号（Scaling Signals）。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式（Integration Pattern）。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管型 Kubernetes（Managed Kubernetes）专属指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — 工作负载变体自动扩缩器（Workload Variant Autoscaler）设计。