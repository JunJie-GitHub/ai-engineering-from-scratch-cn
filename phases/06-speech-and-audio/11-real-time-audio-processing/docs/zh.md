# 实时音频处理 (Real-Time Audio Processing)

> 批处理流水线 (Batch Pipelines) 处理的是整个文件。而实时流水线 (Real-Time Pipelines) 必须在下一个 20 毫秒数据到达前，处理完当前的 20 毫秒数据。每一个对话式 AI (Conversational AI)、广播工作室和电话机器人 (Telephony Bot) 的成败都取决于这一延迟预算 (Latency Budget)。

**类型：** 构建
**语言：** Python, Rust
**前置条件：** 第 6 阶段 · 02（语谱图）、第 6 阶段 · 04（自动语音识别/ASR）、第 6 阶段 · 07（文本转语音/TTS）
**预计耗时：** 约 75 分钟

## 核心问题

你希望打造一个充满“生命力”的语音助手。人类对话中的话轮转换延迟 (Turn-taking Latency) 约为 230 毫秒（从静音到回应）。超过 500 毫秒会显得机械呆板；超过 1500 毫秒则会让人感觉系统已崩溃。到 2026 年，完成一次完整的 **聆听 → 理解 → 响应 → 发声** 循环的延迟预算如下：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 ms |
| 语音活动检测 (VAD) | 10 ms |
| 自动语音识别 (ASR)（流式） | 150 ms |
| 大语言模型 (LLM)（首个 Token） | 100 ms |
| 文本转语音 (TTS)（首个音频块） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai，2024）实现了 200 毫秒的全双工 (Full-duplex) 延迟。GPT-4o-realtime（2024）的延迟约为 320 毫秒。而 2022 年交付的级联流水线 (Cascaded Pipelines) 延迟高达 2500 毫秒。这 10 倍的性能提升主要归功于三项技术：（1）全链路流式处理 (Streaming Everywhere)；（2）基于部分结果的异步流水线 (Asynchronous Pipelining)；（3）可中断生成 (Interruptible Generation)。

## 核心概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**帧 / 块 / 窗口 (Frame / chunk / window)。** 实时音频以固定大小的数据块形式流动。常见选择为 20 毫秒（16 kHz 采样率下为 320 个采样点）。下游所有处理环节都必须跟上这一节奏。

**环形缓冲区 (Ring buffer)。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。避免在热路径（hot path）上进行内存分配。大小 ≈ 最大延迟 × 采样率；例如 2 秒的 16 kHz 环形缓冲区 = 32,000 个采样点。

**语音活动检测 (Voice Activity Detection, VAD)。** 在无人说话时阻断下游处理任务。Silero VAD 4.0（2024）在 CPU 上处理每 30 毫秒帧的耗时不到 1 毫秒。`webrtcvad` 是较旧的替代方案。

**流式自动语音识别 (Streaming ASR)。** 随着音频输入实时输出部分转录文本的模型。流式模式下的 Parakeet-CTC-0.6B（NeMo, 2024）在 320 毫秒延迟下可实现 2%–5% 的词错误率（WER）。Whisper-Streaming（Macháček 等, 2023）对 Whisper 进行分块处理，以约 2 秒的延迟实现近流式识别。

**打断 (Interruption)。** 当用户在助手说话时插话，系统必须：(a) 检测插话行为（barge-in），(b) 停止文本转语音（TTS），(c) 丢弃大语言模型（LLM）剩余的生成内容。所有操作需在 100 毫秒内完成，否则用户会感觉助手“失聪”。

**WebRTC Opus 传输 (WebRTC Opus transport)。** 20 毫秒帧，48 kHz 采样率，自适应码率 8–128 kbps。浏览器和移动端的标准方案。LiveKit、Daily.co 和 Pion 是 2026 年构建语音应用的主流技术栈。

**抖动缓冲区 (Jitter buffer)。** 网络数据包可能乱序或延迟到达。抖动缓冲区负责重新排序并平滑数据流；设置过小会导致可听见的断音，过大则会增加延迟。典型值为 60–80 毫秒。

### 常见陷阱

- **线程竞争 (Thread contention)。** Python 的全局解释器锁（GIL）加上重型模型可能会阻塞音频线程。建议使用基于 C 回调的音频库（如 `sounddevice`、`PortAudio`），并让 Python 远离热路径。
- **采样率转换延迟 (Sample-rate conversion latency)。** 在流水线内部进行重采样会增加 5–20 毫秒延迟。要么在预处理阶段完成重采样，要么使用零延迟重采样器（如 PolyPhase、`soxr_hq`）。
- **TTS 预热 (TTS priming)。** 即使是像 Kokoro 这样快速的文本转语音（TTS）模型，在首次请求时也有 100–200 毫秒的预热时间。应在首次真实对话前缓存模型，并通过一次空跑进行预热。
- **回声消除 (Echo cancellation)。** 若无自动回声消除（AEC），TTS 输出的声音会重新进入麦克风，导致自动语音识别（ASR）误识别机器人自己的语音。WebRTC AEC3 是开源领域的默认方案。

## 动手构建

### 步骤 1：环形缓冲区 (Ring Buffer)

import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)

容量决定了最大缓冲延迟。在 16 kHz 采样率下，32,000 个样本等于 2 秒。

### 步骤 2：语音活动检测门控 (VAD Gate)

def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2

在生产环境中替换为 Silero VAD：

import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5

### 步骤 3：流式自动语音识别 (Streaming ASR)

# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")

### 步骤 4：中断处理器 (Interruption Handler)

class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)

其核心依赖于异步 I/O (Async I/O) 和可取消的文本转语音流 (Cancellable TTS Streaming)。在音频轨道上调用 WebRTC 的 `peerconnection.stop()` 是标准做法。

## 实际应用

2026 年技术栈：

| 层级 | 推荐方案 |
|-------|------|
| 传输层 (Transport) | LiveKit (WebRTC) 或 Pion (Go) |
| 语音活动检测 (VAD) | Silero VAD 4.0 |
| 流式自动语音识别 (Streaming ASR) | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| 大语言模型首字延迟 (LLM First-Token) | Groq、Cerebras 或 vLLM-streaming |
| 流式文本转语音 (Streaming TTS) | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 (Echo Cancel) | WebRTC AEC3 |
| 端到端原生方案 (End-to-End Native) | OpenAI Realtime API 或 Moshi |

## 常见陷阱

- **为了保险起见缓冲 500 毫秒。** 缓冲区*就是*你的延迟下限 (Latency Floor)。尽量缩小它。
- **未绑定线程。** 在优先级低于 UI 的线程上处理音频回调 = 高负载下会出现卡顿/爆音。
- **TTS 分块过小。** 小于 200 毫秒的分块会导致声码器伪影 (Vocoder Artifacts) 可闻。320 毫秒的分块是最佳平衡点。
- **缺少抖动缓冲区 (Jitter Buffer)。** 真实网络存在抖动；若不进行平滑处理，会出现爆音。
- **一次性错误处理。** 音频流水线必须具备防崩溃能力。一个未捕获的异常就会中断整个会话。

## 交付部署

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频流水线，并为每个阶段设定具体的延迟预算。

## 练习

1. **简单（Easy）。** 运行 `code/main.py`。该脚本模拟环形缓冲区（Ring Buffer）与基于能量的语音活动检测（Energy VAD），并打印模拟 10 秒音频流中各阶段的延迟。
2. **中等（Medium）。** 使用 `sounddevice` 构建直通循环（Passthrough Loop），以 20 毫秒帧处理麦克风输入，并在每帧打印语音活动检测（VAD）状态。
3. **困难（Hard）。** 使用 `aiortc` 构建全双工（Full Duplex）回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。使用 1 kHz 脉冲测量端到端延迟（Glass-to-Glass Latency）。

## 关键术语

| 术语 | 常见叫法 | 实际含义 |
|------|-----------------|-----------------------|
| 环形缓冲区（Ring Buffer） | 循环队列 | 用于音频帧的固定大小、无锁（或单生产者单消费者 SPSC 加锁）先进先出（FIFO）队列。 |
| 语音活动检测（VAD） | 静音门控 | 用于区分语音与非语音的模型或启发式规则。 |
| 流式自动语音识别（Streaming ASR） | 实时语音转文本（STT） | 随音频流到达实时输出部分文本；具备有限的前瞻窗口。 |
| 抖动缓冲区（Jitter Buffer） | 网络平滑器 | 对乱序数据包进行重排序的队列；典型延迟为 60–80 毫秒。 |
| 回声消除（AEC） | 回声消除 | 消除扬声器至麦克风的反馈路径信号。 |
| 打断（Barge-in） | 用户中断 | 系统在文本转语音（TTS）播放中途检测到用户语音；必须立即中止播放。 |
| 全双工（Full Duplex） | 双向同时进行 | 用户与机器人可同时发言；Moshi 采用全双工架构。 |

## 延伸阅读

- [Macháček 等人 (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper 模型。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全双工架构，延迟 200 毫秒。
- [LiveKit Agents 框架 (2024)](https://docs.livekit.io/agents/) — 生产级音频智能体（Agent）编排。
- [Silero VAD 仓库](https://github.com/snakers4/silero-vad) — 延迟低于 1 毫秒的 VAD，采用 Apache 2.0 许可证。
- [WebRTC AEC3 论文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除算法。