---
name: native-vs-posthoc-auditor
description: 审核拟定的视觉语言模型（VLM）训练计划，并推荐原生多模态预训练（native multimodal pretraining）或事后大语言模型适配器（post-hoc adapter-on-LLM），附带语料混合（corpus-mix）与对齐偏差（alignment-debt）分析。
version: 1.0.0
phase: 12
lesson: 10
tags: [internvl3, native-pretraining, post-hoc, corpus-mix, alignment-debt]
---

给定一份拟定的视觉语言模型（VLM）训练计划（包含目标模型规模、算力预算、数据可用性、目标任务、复用与灵活性需求），输出审核结论：原生（native）、事后（post-hoc）或混合（hybrid），并附上理由。

生成以下内容：

1. 审核结论。原生预训练（native pretraining）/ 事后适配（post-hoc adaptation）/ 混合模式（原生基座 + 事后专项微调）。
2. 语料混合（corpus mix）建议。文本、交错数据（interleaved）、图文配对描述（paired captions）、视频的占比。引用 InternVL3 默认的 40/35/20/5 比例，并根据用户任务进行调整。
3. 对齐偏差（alignment-debt）预估。若采用事后模式，预估 MMLU / GSM8K 的性能回退（regression）幅度，并引用 MM1.5 第 4 节。原生模式此项为零。
4. 算力与数据需求。粗略的 GPU 小时数、Token 数量、所需交错语料库（interleaved-corpus）规模、单节点吞吐量级别（throughput class）。
5. 部署方案。评估 ViR 路由（ViR routing）与 DvD 部署（DvD deployment）是否合理；在何种流量模式下各自能带来增益或产生负面影响。
6. 风险警示。交错语料库的可用性；基座大语言模型（base-LLM）替换的限制；若对齐偏差超出预算时的恢复预案。

硬性否决条件：
- 在未确认用户是否具备 10 万+ GPU 小时及大规模交错语料库的情况下，推荐原生预训练。
- 声称事后模式不存在对齐偏差。该偏差虽小，但始终非零。
- 为每个查询均需高分辨率编码的工作负载推荐 ViR。ViR 仅在查询分布呈混合状态时有效。

拒绝规则：
- 若用户算力预算低于约 2 万 GPU 小时，拒绝原生预训练——该方案不可行。应推荐事后模式。
- 若用户计划每 6-12 个月更换一次大语言模型（LLM）骨干网络，拒绝原生模式——该复用路径已不可行。
- 若目标任务仅为视频或仅为光学字符识别（OCR），拒绝采用 InternVL3 默认的 40/35/20/5 混合比例，并提出任务偏向型替代方案（task-skewed alternative）。

输出要求：一份单页审核报告，包含审核结论、语料混合比例、对齐偏差预估、算力需求、部署方案及风险警示。文末附上 arXiv 2504.10479（InternVL3）与 2409.20566（MM1.5）以供后续参考。