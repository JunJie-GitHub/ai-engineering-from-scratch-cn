---
name: 语音管道
description: 搭建基于 Pipecat 架构的语音处理管道 (voice pipeline)（VAD + STT + LLM + TTS + 传输层），支持打断 (barge-in)、置信度门控 (confidence gating) 及延迟预算 (latency budget) 控制。
version: 1.0.0
phase: 14
lesson: 22
tags: [语音, pipecat, livekit, webrtc, 延迟]
---

根据语音产品规格说明（语言、传输协议、服务提供商），搭建基于帧 (frame) 的处理管道。

产出内容：

1. `Frame` 类型，包含 `kind`、`payload`、`direction`（downstream / upstream）字段。
2. 处理器 (processors)：`VAD`、`STT`、`LLM`、`TTS`、`Transport`。每个处理器均实现 `process(frame)` 方法。
3. `link()` 辅助函数，用于将处理器按正向和反向顺序串联。
4. 取消帧 (cancel frame) 处理机制：UPSTREAM 路径从 Transport 经 TTS、LLM 到 STT，在各阶段丢弃待处理任务。
5. 观察者 (observers)：各阶段延迟指标；每帧经过处理器时发射一个 OTel span（第 23 课）。
6. STT 置信度门控 (confidence gating)：低于阈值时，发射“请重复”文本帧，而非转录文本。

硬性拒绝条件：

- 管道未实现 UPSTREAM 处理机制。语音交互中，打断 (barge-in) 功能为必选项。
- LLM 调用未采用流式 (streaming) 输出。首字延迟 (first-token latency) 占主导地位，必须使用流式传输。
- STT 缺乏置信度判断。将错误转录文本输入 LLM 会导致生成错误回复。

拒绝规则：

- 若冷启动 (cold run) 端到端延迟超过 1500ms，则拒绝发布。需优化处理链路或改用 MultimodalAgent（LiveKit 直连音频）。
- 若产品以电话通信为主，且管道未集成 SIP 适配器，则拒绝。应通过 LiveKit SIP 或第三方平台（Vapi/Retell）进行路由。
- 若产品传输包含个人身份信息 (PII) 的音频且未启用传输加密，则拒绝。

输出文件：`frames.py`、`processors.py`、`pipeline.py`、`observers.py`、`README.md`（需说明延迟预算、打断设计及传输层选型）。文末需包含“下一步阅读”指引，指向第 23 课（OTel）、第 24 课（可观测性后端 (observability backends)）或 LiveKit 文档以了解 WebRTC 具体细节。