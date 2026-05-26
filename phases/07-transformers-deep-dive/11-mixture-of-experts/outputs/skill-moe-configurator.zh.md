---
name: moe配置器
description: 为新的混合专家模型 (Mixture of Experts, MoE) Transformer 选择专家数量、top-k、负载均衡策略 (Balancing Strategy) 及共享专家布局。
version: 1.0.0
phase: 7
lesson: 11
tags: [Transformer, MoE, 混合专家模型, 模型扩展]
---

给定 Transformer 规格（总参数量预算、每个词元 (Token) 期望的激活参数 (Active Parameters) 数量、可用训练词元数量、推理硬件），输出：

1. MoE 布局 (MoE Layout)。`n_experts`、`top_k`、`n_shared`。前沿规模选择细粒度（256+ 专家，top-8）；较小规模选择经典配置（8 专家，top-2）。附一句理由。
2. 负载均衡策略 (Balancing Strategy)。无辅助损失 (Auxiliary-loss-free)（DeepSeek-V3，默认）、Switch 风格辅助损失，或专家容量与词元丢弃 (Expert Capacity + Token Drop)。若采用无辅助损失，请指明 `γ` 值。
3. 专家并行 (Expert Parallelism) 方案。在给定显存 (VRAM) 条件下，如何将专家分片 (Shard) 至多张 GPU。说明单个专家的显存开销及 GPU 集群总规模。
4. 路由精度 (Routing Precision)。fp32 路由模块 (Router) 得分与 fp16 的对比。在大规模场景下，路由精度至关重要。
5. 故障模式 (Failure Mode) 检查。已命名风险：路由崩溃 (Router Collapse)、专家饥饿 (Expert Starvation)、All-to-All 网络瓶颈、路由开销导致的推理延迟、检查点内存占用 (Checkpoint Memory Footprint)。

若激活参数量低于 4B，拒绝推荐 MoE —— 在同等算力下，稠密架构 (Dense Architecture) 表现更优。对于 2026 年的新项目，拒绝仅使用辅助损失进行负载均衡（无辅助损失已成为默认方案）。若总参数量超过 80 GB，拒绝交付未包含专家并行方案的 MoE 模型。将面向延迟敏感 (Latency-critical) 单用户路径的 MoE 标记为：其速度可能慢于等效的稠密模型。