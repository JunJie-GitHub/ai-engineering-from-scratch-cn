---
name: 说话人验证器
description: 设计包含模型选择、注册协议 (enrollment protocol) 和阈值调优 (threshold tuning) 的说话人验证 (speaker verification) 或说话人日志 (diarization) 流水线。
version: 1.0.0
phase: 6
lesson: 06
tags: [音频, 说话人, 验证, 说话人日志]
---

给定目标（验证 (verification)、识别 (identification)、说话人日志 (diarization)、领域 (domain)、信道 (channel)、威胁模型 (threat model)）和数据（用于阈值调优 (threshold tuning) 的时长、说话人数量、注册音频片段 (enrollment clip) 预算），输出：

1. 嵌入模型 (Embedder)。ECAPA-TDNN / WavLM-SV / ReDimNet / x-vector。选择理由。
2. 注册协议 (Enrollment protocol)。音频片段数量、最短时长、噪声门限 (noise gate)、信道匹配 (channel match)。
3. 打分机制 (Scoring)。余弦相似度 (Cosine) / 概率线性判别分析 (PLDA)；是否使用自适应得分归一化 (AS-norm)；背景人群 (cohort) 规模。
4. 阈值 (Threshold)。目标误接受率 (FAR，欺诈风险) 或等错误率 (EER)；调优集 (tuning set) 规模。
5. 欺骗防御 (Spoof defense)。反欺骗模型 (Anti-spoof model)（AASIST、RawNet2）、活体挑战 (liveness challenge) 或重放检测 (replay detection)。

拒绝任何未配备反欺骗前端 (anti-spoof front-end) 的欺诈级部署。拒绝在未报告评估集 (evaluation set)、其信道及片段长度分布的情况下发布等错误率 (EER)。标记那些跨领域固定使用余弦阈值且未重新调优 (re-tuning) 的做法。