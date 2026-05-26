# 构建语音助手流水线（Pipeline）—— 第 6 阶段综合项目（Capstone）

> 将第 01-11 课的内容串联整合。构建一个能够聆听、推理并语音回复的语音助手。到 2026 年，这已是一个成熟的工程问题，而非研究问题——但集成细节决定了它能否成功交付。

**类型：** 构建实践
**编程语言：** Python
**前置要求：** 第 6 阶段 · 04, 05, 06, 07, 11；第 11 阶段 · 09（函数调用 Function Calling）；第 14 阶段 · 01（智能体循环 Agent Loop）
**预计耗时：** 约 120 分钟

## 问题描述

构建一个端到端（End-to-End）的助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始与结束。
3. 进行流式语音转文本（Speech-to-Text, STT）。
4. 将转录文本传递给支持工具调用（Tool Calling）的大语言模型（Large Language Model, LLM）（如计时器、天气、日历）。
5. 将 LLM 生成的文本流式传输至文本转语音（Text-to-Speech, TTS）引擎。
6. 向用户播放音频。
7. 若用户在回复中途打断，则立即停止。

延迟目标：在笔记本电脑 CPU 上，用户说完话后 800 毫秒内输出首个 TTS 音频字节。质量目标：不遗漏字词、静音片段不产生幻觉字幕、无语音克隆泄露、无提示词注入（Prompt Injection）成功。

## 核心概念

![语音助手流水线：麦克风 → 语音活动检测（VAD） → 语音转文本（STT） → 大语言模型（LLM）+ 工具 → 文本转语音（TTS） → 扬声器](../assets/voice-assistant.svg)

### 七大核心组件

1. **音频采集。** 麦克风 → 16 kHz 单声道 → 20 毫秒数据块。Python 中通常使用 `sounddevice`，生产环境则使用原生 AudioUnit/ALSA/WASAPI。
2. **语音活动检测（Voice Activity Detection, VAD）（第 11 课）。** Silero VAD，阈值设为 0.5，最小语音时长 250 毫秒，静音保持时间（hang-over）500 毫秒。用于触发“开始”与“结束”信号。
3. **流式语音转文本（Streaming STT）（第 4-5 课）。** 可选 Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。输出部分（Partial）与最终（Final）转录结果。
4. **支持工具调用的大语言模型（LLM）。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。使用 JSON Schema 定义工具。流式输出 Token。
5. **流式文本转语音（Streaming TTS）（第 7 课）。** Kokoro-82M（最快的开源模型）或 Cartesia Sonic（商业模型）。在 LLM 输出 20 个 Token 后启动 TTS。
6. **音频播放。** 扬声器输出；针对低带宽网络使用 Opus 编码。
7. **打断处理程序。** 若 TTS 播放期间 VAD 触发，则停止播放、取消 LLM 生成、并重启 STT。

### 你将遇到的三种典型故障模式

1. **首词截断（First-word clip）。** VAD 启动稍慢一拍，导致用户说的“嘿”丢失。建议将起始阈值设为 0.3 而非 0.5。
2. **回复中途打断混乱。** 用户打断后 LLM 仍在继续生成，导致助手与用户声音重叠。需将 VAD 信号直接连接至取消 LLM 生成（cancel-LLM）的逻辑。
3. **静音幻觉（Silence hallucination）。** Whisper 在静音预热帧中输出“感谢观看”等无意义文本。务必始终使用 VAD 进行门控（VAD-gate）。

### 2026 年生产环境参考技术栈

| 技术栈 | 延迟 | 许可证 | 备注 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 毫秒 | 商业 API | 2026 年行业默认方案 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 毫秒 | 大部分开源 | 适合 DIY 开发 |
| Moshi（全双工 Full-duplex） | 200-300 毫秒 | CC-BY 4.0 | 单模型架构；架构不同，见第 15 课 |
| Vapi / Retell（托管服务） | 300-500 毫秒 | 商业 | 上线最快；自定义能力有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线运行 | 开源 | 注重隐私 / 边缘计算 |

## 开始构建

### 步骤 1：带分块处理（Chunking）的麦克风音频采集（伪代码）

import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()

### 步骤 2：基于语音活动检测（Voice Activity Detection, VAD）门控的对话轮次捕获

def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)

### 步骤 3：流式语音转文本（Speech-to-Text, STT）→ 大语言模型（Large Language Model, LLM）→ 文本转语音（Text-to-Speech, TTS）

async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)

### 步骤 4：大语言模型（LLM）循环内的工具调用（Tool Calling）

tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)

### 步骤 5：打断处理（Interruption Handling）

tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break

## 使用方法

请参阅 `code/main.py` 获取一个可运行的模拟程序。该程序使用占位模型（Stub Models）连接了全部七个组件，即使没有硬件也能直观了解流水线（Pipeline）的结构。若要进行实际部署，请将占位模型替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- 用于输入/输出（I/O）的 `sounddevice`

## 常见陷阱

- **永久记录个人身份信息（Personally Identifiable Information, PII）。** 在大多数司法管辖区，完整的对话轮次音频均属于 PII。建议保留期限不超过 30 天，并在静态存储时进行加密。
- **不支持抢话（Barge-in）。** 用户随时可能打断对话。你的助手必须具备停止播报的能力。
- **阻塞式文本转语音（TTS）。** 同步 TTS 会阻塞事件循环（Event Loop）。请使用异步（Async）模式或独立线程。
- **缺乏工具调用（Tool Calling）错误处理。** 工具调用可能会失败。大语言模型（LLM）必须接收错误反馈并尝试重试一次，随后进行优雅降级（Graceful Degradation）。
- **过度严格的幻觉过滤（Hallucination Filters）。** 过滤过严会导致助手反复回复“我无法提供帮助”；过滤过松则可能输出任意内容。应在预留数据集（Held-out Set）上进行校准。
- **缺少唤醒词（Wake-word）选项。** 持续监听会带来隐私风险。建议添加唤醒词门控机制（如 Porcupine 或 openWakeWord）。

## 交付上线

保存为 `outputs/skill-voice-assistant-architect.md`。在预算、规模、语言和合规性约束条件下，输出一份全栈技术规范。

## 练习

1. **简单。** 运行 `code/main.py`。该脚本使用桩模块（Stub Modules）模拟端到端（End-to-End）的完整交互轮次（Turn），并打印各阶段的延迟（Latency）。
2. **中等。** 将语音转文本（Speech-to-Text, STT）桩模块替换为真实的 Whisper 模型，并使用预录制的 `.wav` 音频进行测试。测量词错误率（Word Error Rate, WER）和端到端延迟。
3. **困难。** 添加工具调用（Tool Calling）功能：实现 `get_weather`（可接入任意 API）和 `set_timer`。将大语言模型（Large Language Model, LLM）的请求路由至工具，并验证当用户说出“设置一个 5 分钟的计时器”时，能否正确触发对应函数，且语音回复能予以确认。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 交互轮次 (Turn) | 用户与助手的一次往返交互 | 由语音活动检测（Voice Activity Detection, VAD）界定的一段用户语音，加上一次大语言模型至语音合成（Text-to-Speech, TTS）的响应。 |
| 打断 (Barge-in) | 插话/中断 | 用户在助手说话时开口；助手随即停止播报。 |
| 唤醒词 (Wake Word) | “嘿，助手” | 短关键词检测器；如 Porcupine、Snowboy、openWakeWord。 |
| 端点检测 (End-pointing) | 轮次结束 | 结合语音活动检测（VAD）与最小静音时长，判定用户已结束发言。 |
| 预录缓冲 (Pre-roll) | 语音前缓冲 | 在语音活动检测（VAD）触发前保留 200-400 毫秒的音频，以避免首词被截断。 |
| 工具调用 (Tool Call) | 函数调用 | 大语言模型（LLM）输出 JSON；运行时环境进行分发；结果在循环内回传。 |

## 延伸阅读

- [LiveKit — 语音智能体快速入门](https://docs.livekit.io/agents/) — 生产级参考实现。
- [Pipecat — 语音智能体示例](https://github.com/pipecat-ai/pipecat) — 适合自行搭建的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管式原生语音方案。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工（Full-duplex）参考实现（第 15 课）。
- [Porcupine 唤醒词](https://picovoice.ai/products/porcupine/) — 唤醒词门控机制。
- [Anthropic — 工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — 大语言模型（LLM）函数调用。