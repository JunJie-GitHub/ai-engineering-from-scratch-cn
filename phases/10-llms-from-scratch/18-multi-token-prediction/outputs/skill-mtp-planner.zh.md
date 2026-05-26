---
name: mtp-planner
description: 为新预训练运行规划多词元预测（Multi-Token Prediction）集成方案。
version: 1.0.0
phase: 10
lesson: 18
tags: [mtp, multi-token-prediction, deepseek-v3, pre-training, speculative-decoding]
---

给定预训练运行规格（模型规模、隐藏层维度、层数、数据词元预算、GPU 拓扑结构、目标部署环境）以及明确的目标（更密集的训练信号 vs 投机解码（Speculative Decoding）草稿 vs 两者兼顾），请生成一份多词元预测（Multi-Token Prediction, MTP）集成方案。

输出内容需包含：

1. 深度 D。选择 1 或 2。DeepSeek-V3 采用 D=1，并报告第一层深度的投机解码接受率在 80% 以上。对于大多数训练任务而言，D=2 已进入边际收益递减区间。请结合计算预算论证该选择的合理性——每增加一层深度，每个训练步骤大约会增加一个 Transformer 块（Transformer Block）的计算量。
2. Lambda 调度策略（Lambda Schedule）。默认值：训练前 10% 阶段设为 0.3，之后设为 0.1。对于小型模型（参数量低于 7B），由于更密集的训练信号更为关键，可在训练初期将值上调至 0.5；若观察到 MTP 损失（MTP Loss）主导了主损失（Main Loss），则应下调该值。
3. 参数预算。报告各模块相对于主模型的参数量。确认额外开销低于主模型参数的 5%（稠密架构/Dense）或 3%（混合专家架构/Mixture of Experts, MoE）。
4. 内存与计算开销。量化每个训练步骤额外的前向传播浮点运算次数（FLOPs）（约为 `D * transformer_block_cost`）、额外的反向传播内存（D 个模块的激活内存），以及额外的峰值显存（VRAM）（共享嵌入层（Embedding）和输出头（Head）不计入，投影层（Projection）和 Transformer 块需计入）。
5. 推理阶段集成方案。描述如何在推理时将 MTP 模块作为投机解码草稿进行调用。指明 Leviathan 规则（Leviathan Rule）的集成路径以及 KV 回滚（KV-Rollback）的状态管理机制。确认其与目标推理框架（vLLM、SGLang、TensorRT-LLM）的兼容性。

硬性拒绝条件：
- 为未使用 MTP 预训练的稠密模型添加 MTP。无法后期改造——MTP 模块未经过训练。
- 首次集成时 D > 2。相较于 D=1 收益甚微，且复杂度会急剧上升。
- 在活跃参数量低于 1B 的模型上使用 MTP。在该规模下，信号增益弱于其带来的开销成本。
- 当目标为投机解码时使用并行头（Gloeckle 风格/Gloeckle-style）。它们无法形成因果链式依赖。

拒绝规则：
- 若预训练数据以短序列（长度低于 2k）为主，则拒绝。MTP 的收益前提是序列足够长，使得深度为 2 的监督信号具有实际意义。
- 若目标推理框架完全不支持投机解码，需注明 MTP 仍可提供更密集的训练信号并继续推进，但需标记此不匹配情况。
- 若用户试图在不含 MTP 的现有稠密模型检查点（Checkpoint）上继续预训练，则拒绝，并建议仅在从头开始的全新训练运行或干净的数据边界重置时添加 MTP。

输出要求：一份单页集成方案，需列出 D 值、Lambda 调度策略、参数开销（绝对值与百分比）、计算开销（每个训练步骤的百分比），以及推理阶段的投机解码集成方案。最后需附上一段“成功标准（Success Criterion）”，明确指出用于证明保留 MTP 合理性的实测指标：在训练 50B 词元后，深度 1 的接受率必须高于 70%，否则应回退该架构设计。