# 综合项目 03 — 实时语音助手（ASR 到 LLM 再到 TTS）

> 体验良好的语音代理（Voice Agent）端到端延迟（End-to-End Latency）需低于 800 毫秒，能够准确判断用户何时停止说话，支持打断（Barge-in），并且能在调用工具时不造成音频卡顿。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年均达到了这一标准。它们的架构如出一辙：流式自动语音识别（Streaming ASR）、话轮检测器（Turn-Detector）、流式大语言模型（Streaming LLM）和流式文本转语音（Streaming TTS），全部通过 WebRTC 连接，并在每个传输节点都设定了严格的延迟预算。动手构建一个，测量词错误率（Word Error Rate, WER）、平均意见得分（Mean Opinion Score, MOS）和误截断率（False-Cutoff Rate），并在丢包（Packet Loss）环境下进行测试。

**类型：** 综合项目（Capstone）
**编程语言：** Python（代理与流水线）、TypeScript（Web 客户端）
**前置要求：** 第 6 阶段（语音与音频）、第 7 阶段（Transformer 架构）、第 11 阶段（大语言模型工程）、第 13 阶段（工具调用）、第 14 阶段（智能体）、第 17 阶段（基础设施）
**涉及阶段：** P6 · P7 · P11 · P13 · P14 · P17
**预计耗时：** 30 小时

## 问题描述

语音交互已成为 2025 至 2026 年间发展最迅速的 AI 用户体验（AI UX）领域，其技术门槛每季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 以及 Pipecat 0.0.70 均已将“首段音频输出（First-Audio-Out）延迟低于 800 毫秒”变为现实。但真正的标准不仅仅是延迟，而是交互体验：不随意打断用户、不被系统误打断、能够从句子中途的打断中恢复、在对话过程中调用工具而不阻塞音频流，以及在抖动剧烈的移动网络中保持稳定运行。

仅靠拼接三个 REST 调用（REST Calls）无法实现这一目标。其架构必须是端到端的流水线式流处理（Pipelined Streaming）。当你真正动手构建时，各种故障模式便会暴露无遗：为电话音频调优的语音活动检测器（Voice Activity Detection, VAD）被背景电视声误触发；话轮检测器在等待永远不会出现的标点符号；文本转语音（Text-to-Speech, TTS）引擎在输出前缓冲了 400 毫秒。本综合项目的目标就是在负载压力下逐一解决这些问题，并发布一份延迟与质量评估报告。

## 核心概念

该流水线（Pipeline）包含五个流式处理阶段：**音频输入（Audio In）**（来自浏览器的 WebRTC 或 PSTN）、**自动语音识别（ASR）**（流式输出 Deepgram Nova-3 或 faster-whisper 的部分转录文本）、**话轮检测（Turn Detection）**（结合语音活动检测（VAD）与一个小型话轮检测模型，通过读取部分转录文本来判断完成信号）、**大语言模型（LLM）**（一旦判定话轮结束即开始流式输出 Token）、**文本转语音（TTS）**（在接收到首个 LLM Token 后约 200ms 内开始流式输出音频）。

三个贯穿全局的核心考量。**打断机制（Barge-in）**：当智能体正在说话时用户开始发言，TTS 会立即取消输出，ASR 随即接管。**工具调用（Tool Use）**：对话中途的函数调用（如查询天气、日历）必须在独立侧信道中运行，以免阻塞音频流；若延迟超过 300ms，智能体会预先填充一个确认 Token（如“请稍等……”）。**背压控制（Backpressure）**：在发生丢包时，系统会暂存部分转录文本，VAD 会提高语音门限阈值，智能体也会避免在未收到确认的消息上继续发言。

评估指标均为量化标准。在 15 dB 信噪比（SNR）的 Hamming VAD 基准测试中，词错误率（WER）低于 8%。在 100 次实测通话中，首音频输出延迟的第 50 百分位数（p50）低于 800ms。误截断率（False-cutoff Rate）低于 3%。TTS 的平均意见得分（MOS）高于 4.2。单台 g5.xlarge 实例支持 50 路并发通话。这些数据即为最终交付标准。

## 架构

browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse

## 技术栈

- 传输层（Transport）：LiveKit Agents 1.0（WebRTC）结合 Twilio PSTN 网关；备选框架为 Pipecat 0.0.70
- 自动语音识别（ASR）：Deepgram Nova-3（流式传输，首段部分结果延迟低于 300ms）或自托管的 faster-whisper Whisper-v3-turbo
- 语音活动检测（VAD）：Silero VAD v5 结合 LiveKit 话轮检测器（读取部分转录文本的小型 Transformer 模型）
- 大语言模型（LLM）：OpenAI GPT-4o-realtime（用于深度集成）、Gemini 2.5 Flash Live，或级联的 Claude Haiku 4.5（流式补全，独立音频路径）
- 文本转语音（TTS）：Cartesia Sonic-2（首字节延迟最低）、ElevenLabs Flash v3，或用于自托管的开源模型 Orpheus
- 工具（Tools）：通过 FastMCP 侧信道处理天气/日历/预订请求；若工具调用耗时超过 300ms，智能体将预先输出填充语
- 可观测性（Observability）：OpenTelemetry 语音跨度（spans），Langfuse 语音追踪（traces）并支持音频回放
- 部署（Deployment）：单台 g5.xlarge 实例（24GB 显存）用于自托管 Whisper + Orpheus；使用托管 API 以实现最低延迟

## 构建指南

1. **WebRTC 会话。** 搭建一个 LiveKit 房间及一个用于流式传输麦克风音频的 Web 客户端。在服务器端，挂载一个加入该房间的智能体工作进程（agent worker）。

2. **ASR 流式处理。** 将 20ms 的 PCM 音频帧输入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分（partial）与最终（final）转录结果。记录每次部分结果的延迟。

3. **VAD 与话轮检测器。** 在音频帧流上运行 Silero VAD v5。触发语音结束事件时，将最新的部分转录文本送入 LiveKit 话轮检测器进行判断。仅当 VAD 检测到 500ms 静音且话轮检测器的完成度评分大于 0.6 时，才确认“话轮结束”。

4. **LLM 流式输出。** 话轮结束后，结合当前对话上下文与最终转录文本发起 LLM 调用。流式输出 Token。在生成首个 Token 时，立即将控制权移交至 TTS 模块。

5. **TTS 流式输出。** Cartesia Sonic-2 将音频块流式返回。首个音频块必须在首个 LLM Token 生成后的 200ms 内离开服务器。将音频块发送至 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区（jitter buffer）进行播放。

6. **打断（Barge-in）。** 当 TTS 正在播放时，若 VAD 检测到用户新的语音输入，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新激活 ASR。发布一个 `tts_canceled` 跨度（span）。

7. **工具侧信道。** 将天气和日历注册为函数调用（function-calling）工具。调用触发时并发执行；若 300ms 内未返回结果，则让 LLM 输出“请稍等，我查一下”作为填充语；待工具返回结果后继续执行。

8. **评估测试框架。** 录制 100 通通话。计算词错误率（Word Error Rate, WER，对照预留的转录文本）、误截断率（用户说话中途 TTS 被取消的比例）、首音频输出延迟 p50、TTS 平均意见得分（Mean Opinion Score, MOS，人工评估或 NISQA 模型），以及抖动丢包测试（模拟丢弃 3% 的数据包）。

9. **负载测试。** 使用合成呼叫器在单台 g5.xlarge 实例上驱动 50 个并发通话。测量持续的首音频输出延迟 p95。

## 使用指南

caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out

## 交付上线

`outputs/skill-voice-agent.md` 是最终的交付物。针对特定业务领域（如客户支持、日程安排或自助服务终端），该文件会部署一个 LiveKit 智能体 (Agent)，其自动语音识别 (ASR) / 语音活动检测 (VAD) / 大语言模型 (LLM) / 文本转语音 (TTS) 处理流水线已根据性能指标进行调优。评估标准如下：

| 权重 | 评估标准 | 测量方法 |
|:-:|---|---|
| 25 | 端到端延迟 (End-to-end latency) | 在 100 通录音通话中，p50 首次音频输出 (first-audio-out) 低于 800ms |
| 20 | 话轮转换质量 (Turn-taking quality) | 在 Hamming VAD 基准测试中，误截断率 (False-cutoff rate) 低于 3% |
| 20 | 工具调用正确性 (Tool-use correctness) | 对话中途的工具调用能返回正确数据，且不会导致音频卡顿 |
| 20 | 丢包环境下的可靠性 (Reliability under packet loss) | 注入 3% 丢包率时，词错误率 (WER) 与话轮转换的稳定性 |
| 15 | 评估框架完整性 (Eval harness completeness) | 使用公开配置可实现可复现的测量结果 |
| **100** | | |

## 练习

1. 在 `g5.xlarge` 实例上，将 Deepgram Nova-3 替换为 `faster-whisper v3 turbo`。测量延迟与词错误率 (WER) 的差异，并明确在哪些环节 CPU 与 GPU 的选型决策至关重要。
2. 添加中断仲裁策略 (interruption-arbitration policy)：当用户在工具调用期间插话 (barge-in) 时，智能体应如何应对？对比三种策略（强制取消、完成工具调用后停止、将下一话轮加入队列）。
3. 运行对抗性话轮检测器测试：在用户句子中间设置长停顿。调整 VAD 静音阈值与话轮检测器得分阈值，以在不突破 900ms 延迟上限的前提下实现最低的误截断率。
4. 通过 Twilio 将同一智能体部署至公共交换电话网 (PSTN)。对比 PSTN 与 WebRTC 的首次音频输出延迟。解释抖动缓冲区 (jitter-buffer) 与编解码器 (codec) 的差异。
5. 为非英语语言（日语、西班牙语）添加语音活动检测功能。对比 `Silero VAD v5` 的误触发率与针对特定语言微调后的模型表现。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 话轮检测 (Turn detection) | “话语结束” | 一种分类器，结合 VAD 静音状态与部分转录文本，判断用户是否已停止说话 |
| 插话/打断 (Barge-in) | “中断处理” | 当 VAD 检测到用户新语音时，在播放中途取消 TTS 输出 |
| 首次音频输出 (First-audio-out) | “延迟” | 从用户停止说话到服务器发出第一个音频数据包的时间 |
| 语音活动检测 (VAD) | “语音闸门” | 将音频帧分类为语音或静音的模型；`Silero VAD v5` 是 2026 年的默认选择 |
| 抖动缓冲区 (Jitter buffer) | “音频平滑” | 客户端缓冲区，短暂缓存数据包以吸收网络波动 |
| 填充词 (Filler) | “确认标记” | 当工具调用较慢时，智能体发出的简短短语，用于避免对话出现空白静音 |
| 平均意见得分 (MOS) | “主观评分” | 语音感知质量评级；`NISQA` 是其自动化代理指标 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — WebRTC 智能体（Agent）参考框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 另一种以 Python 优先的流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式自动语音识别（ASR）参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — 语音活动检测（VAD）参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟文本转语音（TTS）参考
- [Retell AI 架构](https://docs.retellai.com) — 生产级语音智能体架构
- [Vapi.ai 生产级技术栈](https://docs.vapi.ai) — 备选生产级参考方案