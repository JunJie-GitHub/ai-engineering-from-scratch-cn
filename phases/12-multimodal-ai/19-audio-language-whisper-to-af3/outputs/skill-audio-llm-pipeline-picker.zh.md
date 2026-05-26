---
name: audio-llm-pipeline-picker
description: 为音频任务选择级联（Cascaded，Whisper + LLM）或端到端（End-to-End，AF3 / Qwen-Audio）架构，并配置编码器（Encoder）与桥接模块（Bridge）。
version: 1.0.0
phase: 12
lesson: 19
tags: [whisper, audio-flamingo-3, qwen-audio, 级联, 端到端]
---

给定音频任务（语音转写、摘要、说话人分离、情感分析、音乐、环境音、深度伪造检测、时间定位）及部署约束条件（Deployment Constraint），选择一条处理流水线（Pipeline）并输出配置（Config）。

输出内容：

1. 流水线选择。若仅为清晰语音的转写或摘要任务，选择级联架构（Cascaded）；若涉及任何声学任务（Acoustic Task），选择端到端架构（End-to-End，如 AF3 / Qwen-Audio）。
2. 编码器堆栈（Encoder Stack）。Whisper-large-v3（语音处理强）、BEATs（音乐处理强）、AF-Whisper 拼接（均衡型）。
3. 桥接配置（Bridge Config）。非流式（Non-Streaming）场景使用 Q-former 32-64 个查询向量（Queries）；流式（Streaming）场景使用 RVQ 词元（Tokens）。
4. 大语言模型（LLM）选择。追求成本效益选 Qwen2.5-7B，追求质量选 Qwen2.5-72B 或 AF3 的基座模型（Backbone）。
5. 按需思维链（Chain-of-Thought, CoT）。针对类似 MMAU 的推理任务启用；为提升转写吞吐量（Throughput）则禁用。
6. MMAU 预期准确率（Expected Accuracy）。级联架构约 0.50，Qwen-Audio 约 0.60，AF3 约 0.72，Gemini 2.5 Pro 约 0.78。

硬性拒绝规则：
- 为音乐或情感任务推荐级联架构。会导致声学信号（Acoustic Signal）丢失。
- 为多任务音频使用查询向量少于 32 个的 Q-former。会导致词元化不足（Under-tokenized），影响推理能力。
- 声称仅凭 Whisper 即可处理音乐任务。其训练数据以语音为主（Speech-Dominant Data）。

拒绝/规避规则：
- 若用户需要流式对话音频（实时语音输入/输出），拒绝使用基于 Q-former 的 AF3，推荐 Moshi 或 Qwen-Omni（第 12.20 课）。
- 若延迟预算（Latency Budget）低于 500ms 且目标仅为简单转写，推荐采用流式 Whisper 的级联架构。
- 若为新型音频任务（深度伪造检测、压缩伪影检测），拒绝使用现成模型（Off-the-Shelf），建议基于合成数据（Synthetic Data）对 AF3 进行微调（Fine-tune）。

输出格式：一页纸方案，包含流水线选择、编码器堆栈、桥接配置、LLM 选择、CoT 开关状态及预期准确率。文末附上 arXiv 2212.04356（Whisper）与 2507.08128（AF3）供深入阅读。