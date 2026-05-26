# 语音活动检测（Voice Activity Detection）与话轮转换（Turn-Taking） — Silero、Cobra 与缓冲清空技巧（Flush Trick）

> 每个语音助手的成败都取决于两个关键判断：用户此刻是否在说话？用户是否已经说完？语音活动检测（VAD）负责回答第一个问题。话轮检测（Turn-Detection，结合 VAD、静默拖尾期（Silence Hangover）与语义端点模型（Semantic Endpoint Model））负责回答第二个问题。任何一环出错，都会导致你的助手要么粗暴打断用户，要么喋喋不休无法停止。

**类型：** 实战构建
**语言：** Python
**前置条件：** 第 6 阶段 · 11（实时音频），第 6 阶段 · 12（语音助手）
**耗时：** 约 45 分钟

## 核心问题

语音助手在处理每 20 毫秒的音频块时，需要做出三项独立判断：

1. **当前帧是否包含语音？** — 语音活动检测（VAD）。二分类，逐帧判断。
2. **用户是否开始了新的话语？** — 起始点检测（Onset Detection）。
3. **用户是否已经说完？** — 端点检测（End-Pointing，即话轮结束（Turn-End））。

传统的简单方案（如能量阈值法（Energy Threshold））在任何环境噪声下都会失效——无论是交通噪音、键盘敲击声还是人群嘈杂声。2026 年的现代解决方案是：Silero VAD（开源、基于深度学习）+ 话轮检测模型（语义端点检测（Semantic Endpointing））+ 经 VAD 校准的静默拖尾期（Silence Hangover）。

## 核心概念

![语音活动检测 (VAD) 级联：能量检测 → Silero → 话轮检测器 → 刷新技巧](../assets/vad-turn-taking.svg)

### 三级 VAD 级联架构

**第一层：能量门控 (Energy Gate)。** 成本最低。将均方根 (RMS) 阈值设为 -40 dBFS。可过滤明显的静音，但会对任何高于该阈值的噪声触发响应。

**第二层：Silero VAD**（2020-2026，MIT 许可证）。100 万参数。基于 6000 多种语言训练。在单 CPU 线程上，每处理 30 毫秒音频块仅需约 1 毫秒。在 5% 的假阳性率 (FPR) 下，真阳性率 (TPR) 达 87.7%。开源默认方案。

**第三层：语义话轮检测器 (Semantic Turn Detector)。** 可使用 LiveKit 的话轮检测模型（2024-2026）或自定义的小型分类器。用于区分“句中停顿”与“发言结束”。依赖语言上下文（语调 + 近期词汇），而非仅依靠静音。

### 关键参数及其默认值

- **阈值 (Threshold)。** Silero 输出概率值；在 &gt; 0.5（默认）或 &gt; 0.3（高灵敏度）时判定为语音。阈值越低 = 首词截断越少，但假阳性越多。
- **最短语音时长 (Minimum Speech Duration)。** 过滤短于 250 毫秒的语音——通常为咳嗽声或椅子摩擦声。
- **静音保持时间/端点检测 (Silence Hangover / End-pointing)。** VAD 状态归零后，需等待 500-800 毫秒再判定为话轮结束。时间过短 → 会打断用户。时间过长 → 响应显得迟钝。
- **预录缓冲区 (Pre-roll Buffer)。** 保留 VAD 触发前 300-500 毫秒的音频。防止“嘿”等起始词被截断。

### 刷新技巧 (Flush Trick，Kyutai 2025)

流式语音转文本 (STT) 模型存在前瞻延迟 (Look-ahead Delay)（Kyutai STT-1B 为 500 毫秒，STT-2.6B 为 2.5 秒）。通常需要在语音结束后等待相应时间才能获取完整转录文本。刷新技巧：当 VAD 触发语音结束信号时，**向 STT 发送刷新 (flush) 信号**以强制立即输出。STT 的处理速度约为实时 4 倍，因此 500 毫秒的缓冲区仅需约 125 毫秒即可处理完毕。

端到端延迟：125 毫秒 VAD + 刷新 STT = 实现对话级低延迟。

### 2026 年 VAD 方案对比

| VAD 方案 | 5% FPR 下的 TPR | 延迟 | 许可证 |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | 商业授权 |
| pyannote 分割模型 | 95% | ~10 ms | 类 MIT |

Silero 是合适的默认选择。Cobra 适用于对合规性/准确率要求更高的升级场景。仅依赖能量的 VAD 在 2026 年的生产环境中已无立足之地。

## 动手构建

### 步骤 1：能量门限检测（Energy Gate）

def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs

### 步骤 2：在 Python 中集成 Silero VAD（语音活动检测，Voice Activity Detection）

from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")

### 步骤 3：话轮结束状态机（Turn-End State Machine）

class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None

### 步骤 4：强制刷新（Flush）机制基础框架

def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)

要使该机制生效，语音转文本（Speech-to-Text, STT）服务（如 Kyutai、Deepgram、AssemblyAI）必须支持 `flush` 操作。Whisper 流式版本不支持此功能——它采用基于块（block-based）的处理方式，且始终等待完整的数据块。

## 实际应用

| 应用场景 | VAD 选型 |
|-----------|-----------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 端侧设备（手机） | Silero VAD ONNX |
| 学术研究 / 说话人日志（Speaker Diarization） | pyannote segmentation |
| 零依赖降级方案 | WebRTC VAD（旧版） |
| 需要高质量的话轮结束检测 | Silero + LiveKit 话轮检测器（Turn Detector）分层组合 |

经验法则：除非万不得已，否则绝不要在生产环境中部署仅依赖能量的 VAD 方案。

## 常见陷阱

- **固定阈值。** 在安静环境下有效，但在嘈杂环境中会失效。建议在设备端进行动态校准，或直接切换至 Silero。
- **静音保持时间（Silence Hangover）过短。** 智能体会在用户说话中途打断。对于日常对话，500-800 毫秒是最佳平衡点。
- **静音保持时间过长。** 会导致系统响应迟钝。建议与目标用户进行 A/B 测试以确定最佳值。
- **缺少预录缓冲区（Pre-roll Buffer）。** 会导致丢失用户最初 200-300 毫秒的音频。务必始终维护一个滚动预录缓冲区。
- **忽略语义端点检测（Semantic Endpointing）。** 像“嗯，让我想想……”这样的话语包含较长的停顿。用户非常反感在思考过程中被打断。建议使用 LiveKit 的 turn-detector 或类似方案。

## 部署上线

保存为 `outputs/skill-vad-tuner.md`。为特定任务场景选择语音活动检测 (Voice Activity Detection) 模型、阈值 (threshold)、静默保持时间 (hangover)、前置缓冲 (pre-roll) 以及话轮检测 (turn-detection) 策略。

## 练习

1. **简单。** 运行 `code/main.py`。该脚本模拟“语音 + 静默 + 语音 + 咳嗽”序列，并测试三个层级的 VAD 模型。
2. **中等。** 安装 `silero-vad`，处理一段 5 分钟的录音，调节阈值以同时最小化首词截断 (first-word clips) 和误触发 (false triggers)。报告精确率 (precision) 与召回率 (recall)。
3. **困难。** 构建一个微型话轮检测器：结合 Silero VAD 与基于最后 10 个词嵌入 (embeddings) 的 3 层多层感知机 (MLP)（使用 `sentence-transformers`）。在人工标注的话轮结束数据集上进行训练，使 F1 分数 (F1 score) 比仅使用 Silero 的方案提升 10%。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 语音活动检测 (VAD) | 语音检测器 | 逐帧二分类判断：当前是否为语音？ |
| 话轮检测 (Turn detection) | 端点检测 (End-pointing) | VAD + 静默保持时间 + 语义端点。 |
| 静默保持时间 (Silence hangover) | 语音后等待时间 | 判定话轮结束前的等待时长；通常为 500-800 毫秒。 |
| 前置缓冲 (Pre-roll) | 语音前缓冲 | 在 VAD 触发前保留 300-500 毫秒的音频。 |
| 刷新技巧 (Flush trick) | Kyutai 方案 | VAD → `flush-STT` → 延迟降至 125 毫秒而非 500 毫秒。 |
| 语义端点 (Semantic endpoint) | “对方是否打算停止？” | 基于词汇内容而非仅依赖静默的机器学习分类器。 |
| 5% 误报率下的真阳性率 (TPR @ FPR 5%) | ROC 曲线工作点 (ROC point) | VAD 标准基准测试指标；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 开源 VAD 的参考实现。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商用精度领先方案。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 实现低于 200 毫秒延迟的工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产环境中的语义端点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 传统基线方案。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 达到说话人日志 (diarization) 级别的音频分割。