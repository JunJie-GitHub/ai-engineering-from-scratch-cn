---
name: VAD 调优器
description: 为语音代理（Voice Agent）选择语音活动检测（VAD）模型、阈值、静音保持（Silence Hangover）、预录（Pre-roll）及话轮检测（Turn Detection）策略。
version: 1.0.0
phase: 6
lesson: 14
tags: [vad, silero, cobra, 话轮检测, 刷新技巧]
---

根据工作负载（消费者级 / 呼叫中心 / 边缘计算 / 无障碍辅助；噪声特征；语言混合情况；延迟要求），输出以下内容：

1. 语音活动检测（VAD）。Silero VAD（默认）· Cobra（商用级精度）· pyannote 分割（说话人日志级）· WebRTC VAD（传统/轻量级）。附一句选择理由。
2. 参数。阈值（0.3-0.5）、最小语音时长（200-300 ms）、静音保持（Silence Hangover，400-800 ms）、预录（Pre-roll，250-500 ms）。
3. 语义话轮检测（Semantic Turn Detection）。启用（使用 LiveKit 话轮检测器或自定义多层感知机（MLP））或不启用。理由需与预期的用户说话模式相关联。
4. 刷新技巧（Flush Trick）。启用（若语音转文本（STT）支持——如 Kyutai / Deepgram）或不启用。预期的延迟节省时间。
5. 防护机制（Guards）。拒绝短于最小时长的语音片段；始终保留预录音频；限制单用户的静音保持覆盖上限；若 VAD 服务宕机则采用故障开放（Fail-open）策略（将所有输入视为语音）。

生产环境中拒绝使用仅基于能量的 VAD（Energy-only VAD）——噪声干扰过大。拒绝将静音保持设为零——会导致频繁打断用户。当有专用的 Silero 可用时，拒绝使用基于 Whisper 的 VAD——速度更慢且精度更低。

示例输入：“用于航空公司改签的呼叫中心交互式语音应答（IVR）。背景嘈杂（机场环境）。英语 + 西班牙语。话轮检测延迟 < 500 ms。”

示例输出：
- VAD：选用 Cobra（商用版）以利用其抗噪优势。若成本过高则回退至 Silero。
- 参数：阈值 0.4（机场环境底噪较高）；最小语音时长 300 ms；静音保持 600 ms（用户在 IVR 交互中常会停顿以核对航班号）；预录 400 ms。
- 语义话轮检测：启用 LiveKit 话轮检测器——句中停顿较常见（例如：“我需要改签我的航班……改到明天”）。
- 刷新技巧：在 Deepgram 流式传输中启用。预期节省：话轮结束延迟从 400 ms 降至 150 ms。
- 防护机制：若无法连接 Cobra/Deepgram 则触发故障开放；记录每次 VAD 触发事件至审计日志以便调优。