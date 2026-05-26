# 音频 Transformer（Audio Transformers）—— Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个“吞食”梅尔频谱图（mel spectrograms）并“开口说话”的视觉 Transformer（Vision Transformer, ViT）。

**类型：** 学习
**编程语言：** Python
**前置知识：** 第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 08（编码器-解码器）、第 7 阶段 · 09（ViT）
**预计耗时：** 约 45 分钟

## 问题背景

在 Whisper（OpenAI, Radford 等人，2022）问世之前，业界最先进的自动语音识别（Automatic Speech Recognition, ASR）技术主要依赖 wav2vec 2.0 和 HuBERT——即自监督特征提取器（self-supervised feature extractors）搭配微调输出头（fine-tuned head）。这类模型虽然质量较高，但依赖昂贵且复杂的数据处理流程，且跨领域泛化能力较差（domain-brittle）。此外，多语言语音识别需要为每个语系单独训练模型。

Whisper 做出了三项关键设计决策：

1. **全量数据训练。** 从互联网上抓取了涵盖 97 种语言的 68 万小时弱标注（weakly-labeled）音频。未使用干净的学术语料库，也未依赖音素（phoneme）标签。
2. **单模型多任务。** 通过任务令牌（task tokens），让同一个解码器联合训练转录、翻译、语音活动检测（Voice Activity Detection, VAD）、语种识别（Language ID）和时间戳预测任务。
3. **标准编码器-解码器 Transformer 架构。** 编码器（Encoder）直接处理对数梅尔频谱图（log-mel spectrograms）。解码器（Decoder）以自回归（autoregressively）方式生成文本令牌（text tokens）。无需声码器（vocoder）、无需连接时序分类（Connectionist Temporal Classification, CTC）损失，也无需隐马尔可夫模型（Hidden Markov Model, HMM）。

最终成果：Whisper large-v3 在面对不同口音、背景噪声以及零干净标注数据的语言时，均表现出极强的鲁棒性（robustness）。到 2026 年，它已成为所有开源语音助手以及大多数商业语音助手的默认语音前端（speech front-end）。

## 核心概念

![Whisper 流水线：音频 → 梅尔频谱 → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 窗口化
音频采样率为 16 kHz。进行截断或填充至 30 秒。计算对数梅尔频谱图（log-mel spectrogram）：80 个梅尔频带（mel bins），步长（stride）为 10 毫秒 → 约 3,000 帧 × 80 个特征。这就是 Whisper 模型所看到的“输入图像”。

### 步骤 2 — 卷积主干（convolutional stem）
两个卷积核大小为 3、步长为 2 的一维卷积层（Conv1D）将 3,000 帧缩减至 1,500 帧。在不显著增加参数量的情况下，将序列长度减半。

### 步骤 3 — 编码器（encoder）
一个包含 24 层（针对 large 版本）的 Transformer 编码器（Transformer encoder），处理 1,500 个时间步。采用正弦位置编码（sinusoidal positional encoding）、自注意力机制（self-attention）和 GELU 前馈神经网络（GELU FFN）。输出维度为 1,500 × 1,280 的隐藏状态（hidden states）。

### 步骤 4 — 解码器（decoder）
一个 24 层的 Transformer 解码器。它自回归地（autoregressively）从字节对编码（BPE）词表中生成词元（tokens），该词表是 GPT-2 词表的超集，并额外包含少量音频专用的特殊词元。

### 步骤 5 — 任务词元（task tokens）
解码器的提示词（prompt）以控制词元开头，用于指示模型执行何种任务：

<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>

或

<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>

模型正是基于此约定进行训练的。你可以通过前缀来控制任务类型。这相当于 2026 年语境下的指令微调（instruction-tuning），只不过应用于语音领域。

### 步骤 6 — 输出（output）
采用宽度为 5 的束搜索（beam search）并结合对数概率阈值（log-prob threshold）。当不存在 `<|notimestamps|>` 词元时，模型会每隔 0.02 秒的音频预测一次时间戳。

### Whisper 模型尺寸

| 模型 | 参数量 | 层数 | d_model | 注意力头数（Heads） | 显存（VRAM, fp16） |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo（2024 年发布）将解码器从 32 层精简至 4 层。解码速度提升 8 倍，且词错误率（WER）回退不到 1 个百分点。正是这种解码速度的突破，使得 Whisper-turbo 成为 2026 年实时语音智能体（voice agents）的默认选择。

### Whisper 不具备的功能

- 不支持说话人分离（diarization，即区分谁在说话）。如需此功能，可搭配 pyannote 使用。
- 原生不支持实时流式处理——30 秒的窗口是固定的。现代封装库（如 `faster-whisper`、`WhisperX`）通过语音活动检测（VAD）与重叠分块技术实现了流式处理。
- 在不借助外部切分的情况下，无法处理超过 30 秒的长文本上下文。但在实际应用中表现良好，因为人类语音转录通常不需要长距离上下文。

### 2026 年技术生态

| 任务 | 模型 | 备注 |
|------|-------|-------|
| 英语自动语音识别（ASR） | Whisper-turbo, Moonshine | Moonshine 在边缘设备上速度快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 支持 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 毫秒延迟目标 |
| 文本转语音（TTS） | Piper, XTTS-v2, Kokoro | 采用编码器-解码器架构，但形态类似 Whisper |
| 音频 + 语言 | AudioLM, SeamlessM4T | 在单一 Transformer 中融合文本词元与音频词元 |

## 开始构建

参见 `code/main.py`。我们并不训练 Whisper —— 我们构建的是对数梅尔频谱图 (log-mel spectrogram) 流水线与任务令牌 (task-token) 提示词格式化器。这些才是你在生产环境中实际会接触的部分。

### 步骤 1：合成音频

生成一个频率为 440 Hz、采样率为 16 kHz 的 1 秒正弦波。共 16,000 个采样点。

### 步骤 2：对数梅尔频谱图（简化版）

完整的梅尔频谱图需要快速傅里叶变换 (FFT)。我们采用一种简化的分帧 (framing) 加逐帧能量 (per-frame energy) 版本，以便在不依赖 `librosa` 的情况下展示该流水线：

def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames

帧长为 25 毫秒，跳步为 10 毫秒。这与 Whisper 的窗口设置一致。出于教学目的，此处使用逐帧能量来替代梅尔频带 (mel bins)。

### 步骤 3：填充至 30 秒

Whisper 始终处理 30 秒的音频块。将频谱图填充（或截断）至 3,000 帧。

### 步骤 4：构建提示词令牌

def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens

这就是完整的任务控制接口。一个由 4 个令牌组成的前缀。

## 使用示例

import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])

更快且兼容 OpenAI 接口的方案：

from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")

**2026 年何时选择 Whisper：**

- 使用单一模型实现多语言自动语音识别 (ASR)。
- 对嘈杂、多样化的音频进行鲁棒的转录。
- 用于 ASR 研究或原型开发——最快的起步方案。

**何时选择其他方案：**

- 边缘端超低延迟流式处理——在同等质量下，Moonshine 优于 Whisper。
- 需要低于 200 毫秒延迟的实时对话式 AI——应选用专用的流式 ASR。
- 说话人分离 (speaker diarization)——Whisper 不支持此功能；需额外集成 pyannote。

## 部署上线

参见 `outputs/skill-asr-configurator.md`。该技能模块会为新的语音应用挑选合适的 ASR 模型、解码参数以及预处理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。验证在 16 kHz 采样率、10 ms 跳步下，1 秒信号的帧数约为 100 帧。30 秒信号则约为 3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的对数梅尔频谱图。验证 80 个梅尔频带的结果与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差范围内一致。
3. **困难。** 实现流式推理 (streaming inference)：将音频切分为 10 秒窗口并设置 2 秒重叠，对每个窗口运行 Whisper，然后合并转录文本。在一段 5 分钟的播客样本上，测量其与单次全量处理 (single-pass) 的词错误率 (word-error rate) 差异。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 梅尔频谱图 (Mel spectrogram) | “音频图像” | 二维表示：一个轴为频率仓 (frequency bins)，另一个轴为时间帧 (time frames)；每个单元格存储对数缩放后的能量值。 |
| 对数梅尔频谱 (Log-mel) | “Whisper 看到的内容” | 经过对数变换的梅尔频谱图；用于近似人类对声音响度的感知。 |
| 帧 (Frame) | “一个时间切片” | 25 毫秒的采样窗口；以 10 毫秒的步长进行重叠。 |
| 任务令牌 (Task token) | “语音提示前缀” | 解码器提示词中的特殊标记，例如 `<|transcribe|>` / `<|translate|>`。 |
| 语音活动检测 (Voice Activity Detection, VAD) | “找出语音部分” | 在自动语音识别 (Automatic Speech Recognition, ASR) 前过滤静音的模块；可大幅降低计算成本。 |
| 连接时序分类 (Connectionist Temporal Classification, CTC) | “连接时序分类” | 经典的自动语音识别损失函数，用于免对齐训练；Whisper 并不使用它。 |
| Whisper-turbo | “小型解码器，完整编码器” | 采用 large-v3 编码器搭配 4 层解码器；解码速度提升 8 倍。 |
| Faster-whisper | “生产环境封装版” | 基于 CTranslate2 的重新实现；支持 int8 量化；比 OpenAI 官方参考实现快 4 倍。 |

## 扩展阅读

- [Radford 等人 (2022). 基于大规模弱监督的鲁棒语音识别](https://arxiv.org/abs/2212.04356) — Whisper 原始论文。
- [OpenAI Whisper 仓库](https://github.com/openai/whisper) — 参考代码与模型权重。阅读 `whisper/model.py` 可在约 400 行代码中自上而下查看一维卷积 (Conv1D) 主干、编码器与解码器的完整结构。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 第 5–6 步所述的束搜索 (beam search) 与任务令牌逻辑均在此处；共 500 行，代码清晰易读。
- [Baevski 等人 (2020). wav2vec 2.0：语音表征自监督学习框架](https://arxiv.org/abs/2006.11477) — 先驱工作；在某些场景下其提取的特征仍达到业界最先进水平 (State-of-the-Art, SOTA)。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产环境封装版本，推理速度比官方参考实现快 4 倍。
- [Jia 等人 (2024). Moonshine：面向实时转录与语音指令的语音识别](https://arxiv.org/abs/2410.15608) — 2024 年推出的边缘设备友好型自动语音识别模型，架构类似 Whisper 但规模更小。
- [HuggingFace 博客 — “使用 🤗 Transformers 微调 Whisper 进行多语言自动语音识别”](https://huggingface.co/blog/fine-tune-whisper) — 标准微调指南，涵盖梅尔频谱图预处理及令牌时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（包含编码器、解码器、交叉注意力机制 (cross-attention) 与文本生成逻辑），与本课程架构图完全对应。