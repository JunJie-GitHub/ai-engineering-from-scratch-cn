# 流式语音到语音（Streaming Speech-to-Speech）—— Moshi、Hibiki 与全双工对话（Full-Duplex Dialogue）

> 2024至2026年重新定义了语音AI（Voice AI）。Moshi 发布了一个单一模型，能够以 200 毫秒的延迟同时实现听与说。Hibiki 则采用分块（chunk-by-chunk）方式进行语音到语音的翻译。两者均摒弃了传统的 ASR（自动语音识别）→ LLM（大语言模型）→ TTS（文本转语音）流水线（pipeline），转而采用基于 Mimi 编解码器（codec）令牌的统一全双工架构。这已成为全新的参考设计（reference design）。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第6阶段 · 13（神经音频编解码器）、第6阶段 · 11（实时音频）、第7阶段 · 05（完整Transformer）
**Time:** 约75分钟

## 问题所在

基于第11课和第12课构建的每个语音智能体（voice agent）都存在约 300-500 毫秒的固有延迟下限（latency floor）：语音活动检测（VAD）触发、语音转文本（STT）处理、大语言模型（LLM）推理、文本转语音（TTS）生成。每个阶段都有其最低延迟。尽管可以进行调优和并行化处理，但流水线（pipeline）的架构形态从根本上限制了性能上限。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果没有流水线会怎样？如果一个模型直接、持续地接收音频输入并输出音频，而将文本仅作为中间的“内心独白”（inner monologue）而非必需的处理阶段，又会怎样？

答案就是**全双工语音到语音（full-duplex speech-to-speech）**。理论延迟为 160 毫秒（80 毫秒 Mimi 帧 + 80 毫秒声学延迟）。在单张 L4 GPU 上的实际延迟为 200 毫秒。这仅为同类最佳流水线语音智能体延迟的一半。

## 核心概念

![Moshi 架构：两个并行的 Mimi 流 + 内心独白（inner monologue）文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入（Inputs）。** 两个 Mimi 编解码器（codec）流，均为 12.5 Hz × 8 个码本（codebooks）：

- 流 1：用户音频（经 Mimi 编码，持续传入）
- 流 2：Moshi 自身的音频（由 Moshi 生成）

**Transformer（变换器）。** 一个 7B 参数（7B-parameter）的时序 Transformer（Temporal Transformer）同时处理这两个流以及一个文本“内心独白（inner monologue）”流。在每个 80 毫秒的时间步中，它会：

1. 读取最新的用户 Mimi 词元（tokens）（8 个码本）。
2. 读取最近生成的 Moshi Mimi 词元（8 个码本，按生成顺序）。
3. 生成下一个 Moshi 文本词元（内心独白）。
4. 生成下一个 Moshi Mimi 词元（通过一个小型深度 Transformer（Depth Transformer）生成 8 个码本）。

所有三个流——用户音频、Moshi 音频、Moshi 文本——均并行运行。Moshi 可以在说话的同时聆听用户；当用户打断时能够自我中断；还能在不打断主要话语的情况下进行附和（back-channel，如“嗯哼”）。

**深度 Transformer（Depth Transformer）。** 在单个帧内，这 8 个码本并非并行预测——它们之间存在码本间依赖关系。一个小型的 2 层“深度 Transformer”会在 80 毫秒内按顺序预测它们。这是自回归（AR）编解码器语言模型（codec LMs）的标准分解方式（VALL-E 和 VibeVoice 也采用此方法）。

### 为什么内心独白文本（inner-monologue text）有帮助

如果没有显式的文本，模型就必须在声学流中隐式地对语言进行建模。Moshi 的核心思路是：强制模型在输出音频的同时输出文本词元。该文本流本质上就是 Moshi 所说内容的逐字稿。这提升了语义连贯性，使得替换语言模型头（language model head）更加容易，并且能免费获得转录文本。

### Hibiki：流式语音到语音翻译（speech-to-speech translation）

采用相同的架构，使用翻译对数据进行训练。持续输入源语言音频，输出目标语言音频。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——转而使用句子级数据结合 GRPO 强化学习（GRPO reinforcement learning）进行延迟优化。

初始支持四个语言对；仅需约 1000 小时数据即可适配新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** — 全双工对话（full-duplex dialogue）（首发支持法语，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同声传译（simultaneous speech translation）
- **Kyutai STT** — 流式自动语音识别（ASR）（支持 500 毫秒或 2.5 秒前瞻（look-ahead））
- **Kyutai Pocket TTS** — 1 亿参数（100M-param）的文本转语音（TTS）模型可在 CPU 上运行（2026 年 1 月）
- **Unmute** — 在公共服务器上整合上述组件的完整流水线（pipeline）

在 L40S GPU 上的吞吐量：支持 64 个并发会话，处理速度为实时 3 倍。

### Sesame CSM —— 同门兄弟

Sesame CSM（2025）采用了类似的理念——以 Llama-3 为骨干网络（backbone），搭配 Mimi 编解码器头。但 CSM 是单向的（接收上下文 + 文本，生成语音），而非全双工。它是目前市场上“语音临场感（voice presence）”最佳的 TTS 模型，但与 Moshi 的全双工能力仍有区别。

### 2026 年性能数据

| 模型 | 延迟 | 使用场景 | 许可证 |
|-------|---------|----------|---------|
| Moshi | 200 毫秒（L4） | 全双工英/法对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法 ↔ 英流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 同上 | 5 个语言对，无需对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 毫秒（TTFA） | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 毫秒 | 闭源，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 毫秒 | 闭源，Google API | 商业 |

## 动手构建

### 步骤 1：接口设计

Moshi 提供了一个 WebSocket 服务器，持续接收 80 毫秒的 Mimi 编码音频块，并返回同样长度的 Mimi 编码音频块。双向通信，持续进行。

import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)

### 步骤 2：全双工 (Full-duplex) 循环

async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)

两个方向同时运行。Python 的 `asyncio` 或 Rust 的 `futures` 是标准的传输机制。

### 步骤 3：训练目标（概念层面）

对于每一个 80 毫秒的帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，随后是 `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内部独白 (Inner Monologue)）；音频在深度 Transformer 内部按码本 (Codebook) 顺序进行预测。

### 步骤 4：Moshi 的优势与局限

Moshi 的优势：

- 在廉价硬件上实现端到端延迟低于 250 毫秒。
- 支持自然的附和反馈 (Back-channels) 与打断。
- 无需编写流水线 (Pipeline) 胶水代码。

Moshi 的局限：

- 工具调用 (Tool Calling)（未针对此进行训练；需要独立的 LLM 路径）。
- 长程推理（Moshi 是约 80 亿参数的对话模型，而非 Claude/GPT-4）。
- 小众话题的事实准确性。
- 大多数企业级生产用例（2026 年仍建议使用流水线架构）。

## 使用指南

| 场景 | 推荐方案 |
|-----------|------|
| 最低延迟的语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示 / 研究 | Moshi、CSM |
| 带工具的企业级智能体 (Agent) | 流水线架构（第 12 课），而非 Moshi |
| 上下文中的自定义文本转语音 (TTS) | Sesame CSM |
| 任意语言的语音到语音 (Speech-to-Speech) 转换 | GPT-4o Realtime 或 Gemini 2.5 Live（商业版） |

## 注意事项

- **工具调用能力有限。** Moshi 是对话模型，而非智能体框架。如需使用工具，请结合流水线架构。
- **特定声音条件控制 (Voice Conditioning)。** Moshi 使用单一训练好的人格/音色；声音克隆需要单独的训练流程。
- **语言覆盖范围。** 法语和英语表现优异；其他语言支持有限。Hibiki-Zero 有所帮助，但仍需训练数据。
- **资源成本。** 完整的 Moshi 会话会独占一个 GPU 插槽；不适合廉价的共享租户 (Shared-tenant) 部署模式。

## 交付部署

保存为 `outputs/skill-duplex-pipeline.md`。针对语音智能体工作负载，选择流水线架构或全双工架构，并说明理由。

## 练习

1. **简单。** 运行 `code/main.py`。它以符号化方式模拟了双流（two-stream）+ 内部独白（inner-monologue）架构。
2. **中等。** 从 HuggingFace 拉取 Moshi，启动服务器，并测试一次对话。测量从用户语音结束到 Moshi 开始响应之间的墙上时钟延迟（wall-clock latency）。
3. **困难。** 使用你在第 12 课中构建的流水线（pipeline）智能体，在 20 条匹配的测试语音上，将其 P50 延迟（P50 latency）与 Moshi 进行对比。撰写报告，分析在何种架构设计下流水线方案依然具有优势。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 全双工（Full-duplex） | 同时听和说 | 同一模型上同时激活两条音频流。 |
| 内部独白（Inner monologue） | 模型的文本流 | Moshi 在输出音频的同时会并行生成文本词元（text tokens）。 |
| 深度 Transformer（Depth transformer） | 码本间预测器（Inter-codebook predictor） | 小型 Transformer，用于在单个 80 毫秒帧内预测 8 个码本（codebooks）。 |
| Mimi | Kyutai 的编解码器（codec） | 12.5 Hz × 8 码本；融合语义与声学特征；为 Moshi 提供底层驱动。 |
| 流式语音到语音（Streaming S2S） | 实时音频转音频 | 逐块（chunk-by-chunk）进行翻译或对话，无需分阶段流水线。 |
| 附和/反馈（Back-channeling） | “嗯嗯”等反应 | Moshi 可在不中断自身发言轮次的情况下发出简短的确认回应。 |

## 延伸阅读

- [Défossez 等人 (2024). Moshi — 语音-文本基础模型](https://arxiv.org/html/2410.00037v2) —— 原始论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) —— 无需对齐数据的流式翻译。
- [Sesame (2025). 跨越语音的恐怖谷](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) —— CSM 规范。
- [Kyutai — Moshi 代码库](https://github.com/kyutai-labs/moshi) —— 安装与服务器部署。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) —— 闭源商业竞品。
- [Kyutai — 延迟流建模](https://github.com/kyutai-labs/delayed-streams-modeling) —— 底层语音转文本/文本转语音（STT/TTS）框架。