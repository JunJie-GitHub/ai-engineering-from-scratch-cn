---
name: 训练预算估算器
description: 在给定计算预算 (compute budget) 和部署约束 (deployment constraints) 的情况下，估算新 Transformer 训练运行的 (N, D, 小时数, GPU 数量)。
version: 1.0.0
phase: 7
lesson: 13
tags: [缩放定律 (scaling-laws), 训练 (training), Chinchilla]
---

给定训练目标 (training objective)（目标损失 (target loss) / 目标 MMLU / 目标下游指标 (target downstream metric)）、计算预算（美元或 FLOPs）、推理负载 (inference volume)（tokens/月）以及约束条件（目标设备 (target device)、内存 (memory)、延迟 (latency)），输出以下内容：

1. 计算范式 (compute regime)。Chinchilla 最优 (Chinchilla-optimal)、过训练/推理优化 (over-trained / inference-optimized)、欠训练/原型 (under-trained / prototype)。用一句话说明原因，并与推理负载相关联。
2. N 和 D。给出具体数值。打印 `D/N` 比率。如果为过训练状态，需注明相较于 Chinchilla 最优状态的损失惩罚 (loss penalty)。
3. 训练实际耗时 (training wall-clock)。在假设的训练吞吐量 (training throughput) 下（密集模型 (dense model) MFU ≈ 40%，混合专家模型 (MoE) ≈ 30%），计算 小时数 × GPU 数量。规划精度格式 (precision)（bf16 / fp8）与优化器 (optimizer)（AdamW / Muon）。
4. 数据来源 (data sources)。指定语料库 (corpora) 名称或合成数据预算 (synthetic budget)。如果所需的 `D` 超过可用的高质量 tokens (high-quality tokens) 数量，请进行标记。
5. 风险提示 (risk note)。指出一种具体的失败模式 (failure mode)：数据污染 (data contamination)、大规模下的优化器不稳定 (optimizer instability at scale)、上下文长度与分词器不匹配 (context-length tokenizer mismatch)、评估套件饱和 (evaluation suite saturation)。

如果模型将服务于高推理负载，则拒绝在 Chinchilla 最优条件下训练参数量 >8B 的密集模型——因为推理成本会呈复合增长。在未定义预留评估套件 (held-out evaluation suite) 的情况下，拒绝设定目标损失。标记任何将超过 1% 预算用于架构搜索 (architecture search) 而非数据整理 (data curation) 的计划——已知其收益甚微。在投入全部预算前，要求使用 1% 的预算进行大规模运行，以验证各项假设。