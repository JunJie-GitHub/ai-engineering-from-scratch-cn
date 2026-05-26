# 语音智能体 (Voice Agents)：Pipecat 与 LiveKit

> 到 2026 年，语音智能体已成为生产环境中的核心类别。Pipecat 提供了一套基于 Python 的基于帧的流水线 (frame-based pipeline)（语音活动检测 (VAD) → 语音转文本 (STT) → 大语言模型 (LLM) → 文本转语音 (TTS) → 传输层 (transport)）。LiveKit Agents 通过 WebRTC 将 AI 模型与用户桥接起来。对于高端技术栈，生产环境的端到端延迟 (latency) 目标通常设定在 450–600ms。

**类型：** 学习
**编程语言：** Python (stdlib)
**前置知识：** 第 14 阶段 · 01（智能体循环 (Agent Loop)），第 14 阶段 · 12（工作流模式 (Workflow Patterns)）
**预计耗时：** 约 60 分钟

## 学习目标

- 描述 Pipecat 的基于帧的流水线：下行 (DOWNSTREAM)（源端→汇端）与上行 (UPSTREAM)（控制流）。
- 列举标准的语音流水线阶段，并说明 Pipecat 支持哪些传输协议。
- 解释 LiveKit Agents 的两个语音智能体类（MultimodalAgent 与 VoicePipelineAgent）及其各自的适用场景。
- 总结 2026 年生产环境对延迟的预期，以及这些预期如何驱动架构选型。

## 核心问题

语音智能体并非简单地在文本循环上拼接文本转语音模块。延迟预算极为苛刻（约 600ms），默认处理的是非完整音频流，话轮检测 (turn detection) 本身就是一个模型，而传输协议则涵盖从传统电话 SIP 到 WebRTC 的多种方案。你要么自行构建基于帧的流水线（如 Pipecat），要么依赖成熟的平台（如 LiveKit）。

## 核心概念

### Pipecat (pipecat-ai/pipecat)

- 基于帧（Frame）的 Python 流水线（Pipeline）框架。
- `Frame` → `FrameProcessor` 链。
- 两个数据流方向：
  - **DOWNSTREAM（下行）** — 源端 → 终端（音频输入，文本转语音（TTS）输出）。
  - **UPSTREAM（上行）** — 反馈与控制（取消、指标、打断（Barge-in））。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）管理生命周期，并提供用于指标/追踪/RTVI 的观察者。

典型流水线：

VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport

传输层（Transport）：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加了结构化对话（状态机（State Machine））功能。Pipecat Cloud 是托管运行时环境。

### LiveKit Agents (livekit/agents)

- 通过 WebRTC 将 AI 模型与用户连接。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两类语音智能体（Voice Agent）：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等效方案实现直接音频交互。
  - **VoicePipelineAgent** — 语音转文本（STT） → 大语言模型（LLM） → 文本转语音（TTS）级联架构；提供文本级控制能力。
- 基于 Transformer 模型的语义话轮检测（Semantic Turn Detection）。
- 原生集成模型上下文协议（MCP）。
- 通过会话初始协议（SIP）支持电话通信。
- 通过 LiveKit Inference 提供 50 多个无需 API 密钥的模型；另有 200 多个模型可通过插件接入。

### 商业平台

Vapi（在优化的高级技术栈上约 450–600ms）和 Retell（在 180 次测试通话中端到端约 600ms）均基于上述框架构建。如果你希望获得托管的语音技术栈，且没有专门的 WebRTC 团队，可以选择这些平台。

### 该模式的常见陷阱

- **缺乏打断（Barge-in）处理。** 用户打断时，智能体仍在继续说话。Pipecat 需要 UPSTREAM 取消帧，LiveKit 也有等效机制。
- **忽略 STT 置信度。** 低置信度的转录文本被直接当作准确内容输入给 LLM。应设置置信度阈值或要求用户确认。
- **TTS 句子中途截断。** 当流水线在语音播报中途取消时，TTS 需要感知该信号或立即切断音频。
- **忽略延迟预算（Latency Budget）。** 每个组件都会增加 50–200ms 的延迟。在发布前务必累加整条链路的延迟。

### 2026 年典型延迟基准

- 语音活动检测（VAD）：20–60ms
- 语音转文本（STT）部分结果：100–250ms
- 大语言模型（LLM）首字延迟（First Token）：150–400ms
- 文本转语音（TTS）首段音频：100–200ms
- 传输层往返时间（RTT）：30–80ms

端到端延迟 450–600ms 属于优质体验。800–1200ms 较为常见。超过 1500ms 则会感觉明显卡顿或故障。

## 动手构建

`code/main.py` 是一个基于帧的示例流水线，包含：

- `Frame` 类型（音频、转录文本、纯文本、TTS 音频、控制指令）。
- 包含 `process(frame)` 方法的 `Processor` 接口。
- 由脚本化处理器组成的五阶段流水线（VAD → STT → LLM → TTS → 传输层）。
- 用于演示打断功能的 UPSTREAM 取消帧。

运行方式：

python3 code/main.py

运行日志将展示正常数据流，以及一次触发打断并中途停止 TTS 播报的取消操作。

## 实际应用

- **Pipecat**：适合需要完全掌控的场景——支持自定义处理器、Python 优先、可插拔的供应商。
- **LiveKit Agents**：适合以 WebRTC 为核心的部署及电话通信场景。
- **Vapi / Retell**：适合没有 WebRTC 团队、希望直接使用托管语音智能体的场景。
- **OpenAI Realtime / Gemini Live**：适合直接音频输入/输出场景（对应 MultimodalAgent）。

## 部署上线

`outputs/skill-voice-pipeline.md` 搭建了一个基于 Pipecat 架构的语音流水线（voice pipeline），包含语音活动检测（VAD）+ 语音转文本（STT）+ 大语言模型（LLM）+ 文本转语音（TTS）+ 传输层（transport），并支持打断处理（barge-in handling）。

## 练习

1. 为你的示例流水线（toy pipeline）添加一个指标观测器（metrics observer）：统计每秒流经各阶段的帧（frame）数量。延迟（latency）主要累积在哪个环节？
2. 实现基于置信度门控的语音转文本（confidence-gated STT）：当置信度低于阈值时，提示“请您再说一遍好吗？”
3. 添加语义话轮检测（semantic turn detection）：采用简单规则——如果转录文本以“?”结尾，则判定为话轮结束。
4. 阅读 Pipecat 的传输层（transport）文档。将标准库传输层替换为 `SmallWebRTCTransport` 配置（占位实现/stub）。
5. 针对同一查询，对比测试 OpenAI Realtime 与 STT+LLM+TTS 级联架构（cascade）的性能。文本级控制（text-level control）会带来多大的延迟成本？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Frame（帧） | “事件（Event）” | 流水线中的强类型数据单元（音频、转录文本、纯文本、控制信号） |
| Processor（处理器） | “流水线阶段（Pipeline stage）” | 包含 `process(frame)` 方法的处理器 |
| DOWNSTREAM（下行） | “正向流（Forward flow）” | 从源头到终端：音频输入，语音输出 |
| UPSTREAM（上行） | “反馈流（Feedback flow）” | 控制流：取消、指标上报、打断 |
| VAD（语音活动检测） | “语音活动检测（Voice activity detection）” | 检测用户何时在说话 |
| Semantic turn detection（语义话轮检测） | “智能话轮结束（Smart end-of-turn）” | 基于模型判断用户是否已说完 |
| MultimodalAgent（多模态智能体） | “直连音频智能体（Direct audio agent）” | 音频进，音频出；中间不经过文本处理 |
| VoicePipelineAgent（语音流水线智能体） | “级联智能体（Cascade agent）” | STT + LLM + TTS 组合；采用文本级控制 |

## 延伸阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的流水线（frame-based pipeline）、处理器（processors）、传输层（transports）
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音原语（voice primitives）
- [Vapi](https://vapi.ai/) — 托管式语音平台（managed voice platform）
- [Retell AI](https://www.retellai.com/) — 托管式语音服务，提供延迟基准测试（latency-benchmarked）