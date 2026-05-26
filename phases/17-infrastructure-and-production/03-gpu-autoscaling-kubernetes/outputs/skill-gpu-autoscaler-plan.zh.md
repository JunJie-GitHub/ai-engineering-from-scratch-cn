---
name: GPU 自动扩缩容方案
description: 为基于 Kubernetes 的大语言模型（Large Language Model, LLM）推理集群设计三层 GPU 自动扩缩容（Autoscaling）方案（Karpenter + KAI Scheduler + 应用信号）。诊断 DCGM_FI_DEV_GPU_UTIL 陷阱与部分分配（Partial Allocation）失败问题。
version: 1.0.0
phase: 17
lesson: 03
tags: [kubernetes, gpu, 自动扩缩容, karpenter, kai-scheduler, hpa, dynamo-planner, llm-d]
---

基于集群拓扑（节点、GPU 类型、NVLink 域）、工作负载特征（张量并行/流水线并行（TP/PP）配置、平均并发数、突发系数）以及服务等级目标（Service Level Objective, SLO）（首字延迟（Time To First Token, TTFT）P99、有效吞吐量（goodput）），制定三层自动扩缩容方案。

输出内容：

1. 第一层 — Karpenter 节点池（NodePool）。指定 `instance-type`、`capacity-type`（按需实例 / 竞价实例 / 预留实例）、`consolidationPolicy`（GPU 池必须设置为 `WhenEmpty`，且 `consolidateAfter: 1h`）、用于排除非 GPU 工作负载的污点（Taints），以及供 KAI Scheduler 选择的标签（Labels）。
2. 第二层 — KAI Scheduler 调度策略。说明是否需要集合调度（Gang Scheduling）（TP/PP > 1 时必须启用）。定义拓扑约束（NVLink 域、机架、可用区）。指定生产环境与训练租户的队列层级及抢占规则。
3. 第三层 — 应用层自动扩缩容器。选择扩缩容信号：预填充（Prefill）密集型工作负载使用队列深度（Queue Depth），解码（Decode）密集型工作负载使用 KV 缓存利用率（KV Cache Utilization），混合型工作负载使用复合有效吞吐量。禁止使用 `DCGM_FI_DEV_GPU_UTIL` 并说明原因。
4. 解耦拆分。若采用第 17 阶段 · 17 课的预填充/解码解耦架构，需指定独立的水平 Pod 自动扩缩容器（Horizontal Pod Autoscaler, HPA）——预填充池使用队列深度信号，解码池使用 KV 缓存利用率信号。
5. 预热池（Warm-Pool）规模设定。针对 SLO 关键路径，基于 P99 TTFT 约束及观测到的冷启动时间（节点供应 + 模型加载），确定最小就绪副本数。
6. 监控。需接入仪表盘的指标：单副本队列深度、单副本 KV 缓存利用率、节点供应等待时间、集合调度延迟次数、Karpenter 整合事件。

硬性拒绝项：
- 建议基于 `DCGM_FI_DEV_GPU_UTIL` 配置 HPA。必须拒绝，并明确指出队列深度与 KV 缓存利用率才是正确的扩缩容信号。
- 为 GPU 池保留 `consolidationPolicy: WhenEmptyOrUnderutilized` 配置。必须拒绝，并引用运行中任务被驱逐（Running-Job-Eviction）的风险。
- 为 TP/PP 工作负载忽略集合调度。必须拒绝——部分分配是一种烧钱反模式（Anti-Pattern）。

拒绝规则：
- 若集群仅包含单一 GPU 类型和单个节点，则拒绝提议使用 Karpenter——客户应优先采用托管无服务器架构（Managed Serverless）（第 17 阶段 · 02 课）。
- 若运维人员要求“基于 GPU 内存进行扩缩容”，必须拒绝——vLLM 会预先分配至 `--gpu-memory-utilization`；即使仅有一个请求，内存占用也会维持在 90% 左右。
- 若以复杂性为由拒绝为 TP-8 工作负载启用集合调度，则拒绝认证该方案——将单个 Pod 调度至 8 块分散的 GPU 上会导致原子性失败。

输出要求：一份单页方案，包含 Karpenter YAML 代码片段、KAI Scheduler 配置片段、HPA/自定义自动扩缩容器信号选择、预热池数量以及五项仪表盘指标。结尾需设置单一熔断开关（Kill-Switch）：若 P99 TTFT 突破阈值，则回滚至最近已知的自动扩缩容状态。