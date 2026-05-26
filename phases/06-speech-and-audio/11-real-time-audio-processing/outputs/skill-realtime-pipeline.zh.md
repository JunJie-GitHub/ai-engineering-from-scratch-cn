---
name: 实时语音流水线
description: 根据目标端到端延迟，选择传输协议、语音活动检测、流式语音转文本、大语言模型、流式文本转语音及编排框架。
version: 1.0.0
phase: 6
lesson: 11
tags: [语音智能体, livekit, pipecat, silero, 流式处理, 延迟]
---

根据给定目标（延迟 P50/P95、语言、通信渠道、本地部署与云端部署、通话量），输出以下内容：

1. 传输协议 (Transport)。WebRTC (LiveKit / Daily) · WebSocket · SIP 中继 (Twilio / Telnyx)。选择依据需与抖动容忍度 (Jitter Tolerance) 及具体用例挂钩。
2. 语音活动检测 (VAD) 与话轮切换 (Turn-taking)。Silero VAD（开源，99.5% 真正例率 TPR）· Cobra（商业版）· LiveKit 话轮检测器。需配置阈值 (Threshold)、最小语音时长及静音拖尾时间 (Silence Hang-over)。
3. 流式语音转文本 (Streaming STT)。Parakeet TDT（最快开源）· Kyutai STT（含强制刷新技巧 Flush Trick）· Deepgram Nova-3（API，约 150 毫秒）· Whisper-streaming。需说明选择理由。
4. 大语言模型 (LLM) 与流式处理。在文本转语音 (TTS) 启动前锁定前 20 个词元 (Token)。需明确模型、流式配置及针对提示词注入 (Prompt Injection) 的安全护栏。
5. 流式文本转语音 (Streaming TTS)。Kokoro-82M（首字延迟 Time To First Audio, TTFA 约 100 毫秒）· Orpheus · Cartesia Sonic · ElevenLabs Turbo。需配置语音包或克隆防护（参考第 8 课）。
6. 编排框架 (Orchestration)。LiveKit Agents · Pipecat · Vapi · Retell · 自研 Rust 方案。选择依据需与团队技术栈及业务规模挂钩。
7. 可观测性 (Observability)。各阶段延迟的 P50/P95/P99 直方图；误打断率；通话中断率；通话样本的词错误率 (Word Error Rate, WER)。

拒绝在 STT 前缓冲完整语句的部署方案。拒绝不支持流式输出的 TTS 方案。拒绝仅使用平均延迟进行评估——必须要求 P95 延迟指标。对于月通话量超过 10 万分钟的场景，若未提供与自研方案的成本对比，则拒绝使用托管平台（Vapi / Retell）。

示例输入：“用于汽车保险报价的语音智能体。P95 延迟 < 500 毫秒。语言：美式英语。通话量：5 万分钟/周。合规要求：类 HIPAA 标准（日志中不得包含个人身份信息 Personally Identifiable Information, PII）。”

示例输出：
- 传输协议：LiveKit Agents + Twilio SIP。已在呼叫中心规模得到验证，支持 HIPAA 模式按需启用。
- VAD：Silero VAD，阈值 0.45，最小语音时长 220 毫秒，静音拖尾 400 毫秒。叠加 LiveKit 话轮检测器。
- STT：Deepgram Nova-3 英语模型（P95 延迟约 150 毫秒）；若需本地审计，则降级使用 Parakeet-TDT。
- LLM：通过 OpenAI 实时 API 流式调用 GPT-4o；使用后过滤器防范提示词注入；将前 20 个词元锁定至 TTS。
- TTS：Cartesia Sonic 2（TTFA 约 150 毫秒，不使用语音克隆——采用预定义音色）。
- 编排框架：LiveKit Agents。生产环境通过 Hamming AI 实现可观测性。
- 日志：在持久化存储前，通过正则表达式与命名实体识别 (Named Entity Recognition, NER) 过滤掉 CVV、SSN 及出生日期 (DOB)。保留 30 天。