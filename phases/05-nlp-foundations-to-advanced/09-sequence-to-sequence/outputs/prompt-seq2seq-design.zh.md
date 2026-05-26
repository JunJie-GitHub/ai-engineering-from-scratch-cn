---
name: seq2seq-design
description: 为给定任务设计序列到序列（sequence-to-sequence）流水线。
phase: 5
lesson: 09
---

给定一项任务（翻译、摘要、改写、问题重写），请输出以下内容：

1. 架构（Architecture）。默认采用预训练 Transformer 编码器-解码器（Transformer encoder-decoder）模型（BART、T5、mBART、NLLB）。仅在特定约束条件（如流式处理、边缘端推理、教学演示）下，才使用基于 RNN 的序列到序列（seq2seq）模型。
2. 初始检查点（Starting checkpoint）。指定模型名称（`facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`）。确保检查点与任务需求及语言覆盖范围相匹配。
3. 解码策略（Decoding strategy）。贪心解码（Greedy decoding）用于生成确定性输出，束搜索（Beam search，宽度 4-5）用于提升质量，带温度参数的采样（Sampling with temperature）用于增加多样性。需提供一句话的理由说明。
4. 上线前需验证的一种故障模式（Failure mode）。暴露偏差（Exposure bias）在较长输出中会表现为生成漂移（Generation drift）；请采样 20 个长度处于第 90 百分位的输出结果进行人工目检（Eyeball）。

若平行语料（Parallel examples）不足约 100 万条，应拒绝推荐从零开始训练序列到序列模型。对于面向用户的内容，任何使用贪心解码的流水线都应标记为脆弱（Fragile），因为贪心解码极易产生重复和死循环。