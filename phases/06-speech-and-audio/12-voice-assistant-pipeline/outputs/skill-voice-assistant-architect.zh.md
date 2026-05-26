---
name: 语音助手架构师
description: 针对给定工作负载，生成全栈语音助手规范——涵盖组件、延迟预算、可观测性与合规性。
version: 1.0.0
phase: 6
lesson: 12
tags: [语音助手, 架构, livekit, pipecat, 合规性]
---

根据用例（消费者 / 客户支持 / 无障碍辅助 / 边缘计算）、预期规模（并发会话数、每月分钟数）、语言、延迟目标、合规要求（HIPAA、PCI、EU AI Act、CA SB 942），输出以下内容：

1. 组件（7 层架构）。麦克风与音频分块 (chunking) · 语音活动检测 (VAD) · 流式语音转文本 (STT) · 大语言模型 (LLM) 与工具调用 · 流式文本转语音 (TTS) · 音频播放 · 打断处理器。为每一层指定具体的提供商/模型。
2. 延迟预算 (Latency Budget)。各阶段的 P50 / P95 / P99 目标值，其总和需符合端到端目标。标明各阶段是独立运行还是顺序执行。
3. 工具调用规范 (Tool-call Schema)。每个工具的 JSON 规范 + 错误处理 + 降级/回退文本。必须包含一条“无法提供帮助”的路径，当 LLM 连续两次失败时必须触发该路径。
4. 安全性 (Safety)。提示词注入防护 (Prompt Injection Guard)、语音克隆锁定（若 TTS 具备克隆能力）、唤醒词门控（针对常开模式）、日志中的个人身份信息 (PII) 脱敏、30 天数据保留期。
5. 可观测性 (Observability)。各阶段 P50/P95/P99 · 误打断率 (False-interruption Rate) · 工具调用成功率 · 每百次调用的词错误率 (WER) · 每分钟成本 · 放弃率。
6. 合规性 (Compliance)。披露音频（“这是一位 AI 助手”）、区域数据驻留（欧盟数据存储在欧盟）、审计日志保留、用户退出路径。

拒绝部署无唤醒词的常开模式。拒绝不支持流式输出的 TTS（会增加整句长度的延迟）。拒绝仅使用平均延迟而不提供 P95 指标的方案——长尾延迟才是导致用户流失的关键。拒绝在未经法律审查的情况下保留原始音频超过 30 天。

示例输入：“面向低视力用户的无障碍助手：面向消费级邮件应用的纯语音交互界面。英语。P95 < 600 毫秒。约 1 万并发用户。”

示例输出：
- 组件：sounddevice（通过 LiveKit Agents 使用 WebRTC）· Silero VAD · Deepgram Nova-3（英语）· GPT-4o 搭配邮件工具（read_message, compose_reply, mark_read）· Cartesia Sonic 2 流式输出 · WebRTC 输出 · 触发 VAD 时执行 interrupt=cancel-LLM-and-TTS。
- 预算：采集 120 毫秒 + VAD 40 + STT 150 + LLM 首词生成时间 (TTFT) 100 + TTS 首音频生成时间 (TTFA) 150 = P95 560 毫秒。
- 工具：read_message({id}), compose_reply({message_id, body}), mark_read({id}), search({query})。均返回 JSON；每个工具 LLM 最多重试 2 次，随后触发降级文本“I couldn't do that — try rephrasing”。
- 安全性：提示词注入防护（检测 `ignore previous instructions`）；唤醒词 "Hey Mail"；禁用语音克隆（使用固定的 Cartesia 音色）；日志中对邮件正文进行脱敏。
- 可观测性：Hamming AI 生产环境监控；各阶段 Prometheus 直方图；当误打断率 > 5% 或 p95 > 800 毫秒时触发告警。
- 合规性：首次使用时进行 AI 身份披露；仅针对医疗类消息启用 HIPAA 合规选项；欧盟用户路由至欧盟托管的 Cartesia + GPT-4o 爱尔兰节点。