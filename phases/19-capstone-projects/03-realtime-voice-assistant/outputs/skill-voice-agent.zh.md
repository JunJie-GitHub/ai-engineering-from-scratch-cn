---
name: voice-agent
description: 构建一个首段音频输出（First-Audio-Out）延迟低于800毫秒、支持打断（Barge-in）处理及对话中工具调用（Tool Calls）的实时语音智能体（Voice Agent）。
version: 1.0.0
phase: 19
lesson: 03
tags: [综合项目, 语音, WebRTC, LiveKit, Pipecat, 自动语音识别(ASR), 文本转语音(TTS), 流式处理]
---

针对特定领域（如客户支持、日程安排、零售助手），部署一个 WebRTC 语音智能体，确保端到端首段音频输出延迟低于 800 毫秒，同时妥善处理用户打断、工具调用及丢包（Packet Loss）问题。

构建计划：

1. 搭建一个 LiveKit Agents 1.0 房间，并配置一个可流式传输麦克风音频的 Web 客户端。接入 Twilio PSTN 网关以支持电话网络覆盖。
2. 运行流式自动语音识别（Streaming ASR）（使用托管的 Deepgram Nova-3 或在 g5.xlarge 实例上部署 faster-whisper 的 Whisper-v3-turbo）。订阅部分（Partial）和最终（Final）转录文本。
3. 在 20 毫秒音频帧上运行 Silero 语音活动检测（VAD）v5。在语音结束时，使用 LiveKit 话轮检测器（Turn-Detector）对最新的部分转录文本进行评分；仅当 VAD 静音时长 >= 500 毫秒且完成度评分 >= 0.6 时，才确认话轮结束（Turn-Complete）。
4. 流式传输大语言模型（LLM）输出（如 GPT-4o-realtime、Gemini 2.5 Flash Live 或级联的 Claude Haiku 4.5）。在 200 毫秒内将首个词元（Token）传递给文本转语音（TTS）模块。
5. 流式传输 TTS（使用 Cartesia Sonic-2 或 ElevenLabs Flash v3）。首个音频块（Audio Chunk）必须在接收到首个 LLM Token 后的 200 毫秒内离开服务器。
6. 打断处理：当 VAD 在系统处于 SPEAKING（播报中）或 THINKING（思考中）状态时检测到新的用户语音，立即取消 TTS，丢弃剩余的 LLM 输出，并重新激活 ASR。发布一个 `tts_canceled` 跨度（Span）。
7. 工具侧信道（Tool Side-Channel）：并发执行函数调用；若延迟超过 300 毫秒，则发送确认填充音（Acknowledgment Filler），确保音频流永不卡顿。
8. 录制 100 通通话。对照预留测试集转录文本测量词错误率（WER），在 Hamming VAD 基准测试上评估误截断率（False-Cutoff Rate），统计首段音频输出延迟的 p50 值、NISQA 平均意见得分（MOS），以及注入 3% 丢包率时的系统表现。
9. 使用合成呼叫者（Synthetic Caller）在单台 g5.xlarge 实例上进行 50 路并发通话的负载测试；报告持续运行下的首段音频输出延迟 p95 值。

评估标准：

| 权重 | 评估标准 | 测量方法 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 通录制通话中，首段音频输出延迟 p50 低于 800 毫秒 |
| 20 | 话轮切换质量 | 在 Hamming VAD 基准测试中，误截断率低于 3% |
| 20 | 工具调用正确性 | 对话中途的工具调用能返回正确数据，且不导致音频流卡顿 |
| 20 | 丢包环境下的可靠性 | 注入 3% 丢包率时，WER 与话轮切换保持稳定 |
| 15 | 评估框架完整性 | 使用公开配置可实现可复现的测量结果 |

硬性否决项：

- 非流式处理管线（批处理 ASR、批处理 TTS）无法达到延迟目标。
- 任何未立即取消 TTS 缓冲区的打断策略。延迟取消会导致最严重的用户体验倒退。
- 同步阻塞 LLM 流的工具调用。此类调用必须在侧信道中运行。

拒绝规则：

- 若未部署 VAD 或话轮检测器，则拒绝上线。固定超时机制的话轮切换会产生不可接受的截断率。
- 若未注明是人工评分还是 NISQA 代理评分，则拒绝报告 MOS 分数。
- 若未提供至少 100 通录制通话记录并公开通话追踪数据（Call Traces），则拒绝报告“低于 X 的 p50 延迟”。

交付物：一个代码仓库，需包含 LiveKit 智能体工作进程（Agent Worker）、PSTN 网关配置、100 通通话评估框架（Eval Harness）、公开的 Langfuse 语音仪表盘、与一家托管竞品（Retell、Vapi 或直接使用 OpenAI Realtime API）的横向对比报告，以及一份详细文档，记录你观察到的三次最严重的话轮切换失败案例，以及修复每个案例所采用的检测器调优参数。