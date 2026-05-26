# 文本转语音（Text-to-Speech, TTS）—— 从 Tacotron 到 F5 与 Kokoro

> 自动语音识别（Automatic Speech Recognition, ASR）将语音转为文本；文本转语音（Text-to-Speech, TTS）则将文本转为语音。2026 年的技术栈（stack）包含三个部分：文本 → 词元（tokens），词元 → 梅尔频谱（mel），梅尔频谱 → 波形（waveform）。每个部分都配有可在笔记本电脑上运行的默认模型。

**类型：** 构建
**语言：** Python
**前置要求：** 第 6 阶段 · 02（频谱图与梅尔频谱），第 5 阶段 · 09（序列到序列模型，Seq2Seq），第 7 阶段 · 05（完整 Transformer，Full Transformer）
**耗时：** 约 75 分钟

## 问题描述

假设你有一段文本：“Please remind me to water the plants at 6 pm.” 你需要生成一段 3 秒的音频片段，要求听起来自然、具备正确的韵律（prosody，如停顿和重音）、准确发音“plants”中的元音，并且能在 CPU 上以低于 300 毫秒的延迟运行，以满足实时语音助手的需求。此外，你还需要支持音色切换、处理语码转换（code-switching）输入（例如“remind me at 6 pm, daijoubu?”），并确保人名发音准确无误。

现代 TTS 流水线（pipeline）通常如下所示：

1. **文本前端（Text frontend）。** 对文本进行规范化处理（如日期、数字、电子邮件），将其转换为音素（phonemes）或子词词元（subword tokens），并预测韵律特征。
2. **声学模型（Acoustic model）。** 文本 → 梅尔频谱（mel spectrogram）。代表模型包括 Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）和 Kokoro（2024）。
3. **声码器（Vocoder）。** 梅尔频谱 → 波形（waveform）。代表模型包括 WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022），以及 2024 年之后出现的神经编解码声码器（neural codec vocoders）。

到了 2026 年，随着端到端扩散模型（diffusion models）和流匹配模型（flow-matching models）的兴起，声学模型与声码器之间的界限逐渐模糊。但在调试时，将系统划分为这三个部分的思维模型（mental model）依然适用。

## 核心概念

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017)。** 序列到序列（Seq2seq）架构：字符嵌入（char-embedding）→ 双向长短期记忆网络（BiLSTM）编码器 → 位置敏感注意力机制（location-sensitive attention）→ 自回归（autoregressive）LSTM 解码器生成梅尔频谱帧（mel frames）。推理速度较慢（受自回归特性限制），处理长文本时韵律易出现波动。目前仍常被作为基线模型引用。

**FastSpeech 2 (2020)。** 非自回归（Non-autoregressive）架构。时长预测器（Duration predictor）负责输出每个音素（phoneme）应分配的梅尔频谱帧数。仅需单次前向传播（1-pass），速度较 Tacotron 快 10 倍。虽因单调对齐（monotonic alignment）损失了部分自然度，但已成为业界广泛部署的标准方案。

**VITS (2021)。** 采用变分推断（variational inference）技术，端到端（end-to-end）联合训练编码器、基于流的时长预测模块与 HiFi-GAN 声码器（vocoder）。具备高质量输出与单模型架构优势。在 2022 至 2024 年间主导了开源语音合成（Text-to-Speech, TTS）领域。主要变体包括：YourTTS（支持多说话人零样本学习 zero-shot）、XTTS v2（2024 年发布，Coqui 出品）。

**F5-TTS (2024)。** 基于流匹配（flow matching）的扩散 Transformer（Diffusion Transformer）架构。韵律表现自然，仅需 5 秒参考音频即可实现零样本声音克隆。在 2026 年开源 TTS 排行榜中位列榜首。参数量为 3.35 亿（335M）。

**Kokoro (2024)。** 模型轻量（8200 万参数），支持在 CPU 上运行，是实时应用场景中表现最佳的英语语音合成模型。仅支持封闭词表的英语，采用 Apache-2.0 开源许可证。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 代表当前商业领域的最先进水平（State of the Art, SOTA）。ElevenLabs v2.5 提供的情感控制标签（如 `[whispered]`、`[laughing]`）及角色语音功能，在 2026 年已主导有声书制作市场。

### 声码器（Vocoder）演进

| 时期 | 声码器 | 延迟 | 音质 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 仅支持离线 | 发布时为最先进水平（SOTA） |
| 2018 | WaveRNN | 接近实时 | 良好 |
| 2020 | HiFi-GAN | 100 倍实时速度 | 接近人类水平 |
| 2022 | BigVGAN | 50 倍实时速度 | 跨说话人/语言泛化能力强 |
| 2024 | SNAC, DAC（神经编解码器 neural codecs） | 与自回归（AR）模型集成 | 离散令牌（discrete tokens），比特效率高 |

截至 2026 年，大多数“TTS”模型已实现从文本到音频波形的端到端生成；梅尔频谱图（mel spectrogram）仅作为模型内部的中间表示。

### 评估指标

- **MOS（平均意见得分 Mean Opinion Score）。** 采用 1-5 分制，通过众包方式采集。目前仍是评估音质的黄金标准，但数据采集过程极其耗时。
- **CMOS（比较平均意见得分 Comparative MOS）。** 采用 A/B 对比偏好测试。单次标注的置信区间更窄，统计效率更高。
- **UTMOS、DNSMOS。** 无参考（reference-free）的神经 MOS 预测模型。广泛用于各类排行榜评估。
- **CER（字符错误率 Character Error Rate）结合自动语音识别（ASR）。** 将 TTS 生成的音频输入 Whisper 模型进行转写，并计算其与原始输入文本的 CER。作为语音可懂度（intelligibility）的代理指标。
- **SECS（说话人嵌入余弦相似度 Speaker Embedding Cosine Similarity）。** 用于量化声音克隆的相似度与质量。

以下为 2026 年在 LibriTTS test-clean 数据集上的测试数据：

| 模型 | UTMOS | CER（通过 Whisper） | 模型大小 |
|-------|-------|-------------------|------|
| 真实音频（Ground truth） | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 动手构建

### 步骤 1：音素化 (phonemize) 输入

from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'

音素 (phoneme) 是通用的桥梁。避免将原始文本直接输入到质量低于 VITS 级别的模型中。

### 步骤 2：运行 Kokoro（2026 年 CPU 默认方案）

from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000

支持离线运行，单文件部署，参数量为 8200 万 (82M)。

### 步骤 3：运行 F5-TTS 进行声音克隆 (voice cloning)

from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)

传入一段 5 秒的参考音频及其对应文本；F5-TTS 即可克隆其韵律 (prosody) 和音色 (timbre)。

### 步骤 4：从零构建 HiFi-GAN 声码器 (vocoder)

代码量过大，无法完整放入教程脚本中，但其基本结构如下：

class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform

训练方式：对抗训练 (adversarial training)（在短时窗上训练判别器）+ 梅尔频谱图 (mel-spectrogram) 重建损失 + 特征匹配 (feature-matching) 损失。该模块已高度标准化——建议直接使用 `hifi-gan` 仓库或 NVIDIA NeMo 提供的预训练权重 (checkpoints)。

### 步骤 5：完整流水线 (pipeline)（伪代码）

text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)

## 实际应用指南

2026 年技术栈推荐：

| 应用场景 | 推荐方案 |
|-----------|------|
| 实时英语语音助手 | Kokoro (CPU) 或 XTTS v2 (GPU) |
| 基于 5 秒参考音频的声音克隆 | F5-TTS |
| 商业化角色语音 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 (fine-tune) |
| 低资源语言 | 使用 5–20 小时目标语言数据训练 VITS |
| 富有表现力/情感标签控制 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

截至 2026 年的开源领跑者：**追求质量选 F5-TTS，追求效率选 Kokoro**。除非你是做技术史研究的，否则别再碰 Tacotron 了。

## 常见陷阱

- **缺少文本正则化器 (text normalizer)。** “Dr. Smith” 会被读作 “Doctor” 还是 “Drive”？“2026” 是读作 “twenty twenty six” 还是 “two zero two six”？务必在音素化之前完成文本正则化。
- **未登录词 (OOV) 专有名词。** “Ghumare” 会被转成 “ghyu-mair” 吗？请为未知词元 (token) 配备一个备用的字素到音素 (grapheme-to-phoneme) 转换模型。
- **音频削波 (clipping)。** 声码器输出通常不会削波，但推理时的梅尔频谱 (mel) 缩放不匹配可能导致波形超出 ±1.0 范围。务必始终使用 `np.clip(wav, -1, 1)` 进行截断。
- **采样率 (sample-rate) 不匹配。** Kokoro 输出为 24 kHz；若下游流水线期望 16 kHz → 必须进行重采样 (resample)，否则会出现混叠失真 (aliasing)。

## 部署上线

保存为 `outputs/skill-tts-designer.md`。为指定的音色、延迟和语言目标设计一个 TTS（Text-to-Speech，文本转语音）流水线。

## 练习

1. **简单。** 运行 `code/main.py`。从示例词表（Toy Vocabulary）构建音素（Phoneme）字典，估算每个音素的持续时间，并打印一个模拟的“梅尔频谱（Mel-spectrogram）”调度表。
2. **中等。** 安装 Kokoro，分别使用 `af_bella` 和 `am_adam` 音色合成同一句话。对比音频时长与主观听感质量。
3. **困难。** 录制一段 5 秒的本人参考音频。使用 F5-TTS 进行声音克隆。报告参考音频与克隆输出之间的 SECS（Speaker Embedding Cosine Similarity，说话人嵌入余弦相似度）值。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 音素（Phoneme） | 声音单元 | 抽象的声音类别；英语中共有 39 个（基于 ARPABet）。 |
| 时长预测器（Duration Predictor） | 每个音素持续多久 | 非自回归（Non-AR）模型的输出；表示每个音素对应的整数帧数。 |
| 声码器（Vocoder） | 梅尔频谱 → 波形 | 将梅尔频谱映射为原始音频样本的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于生成对抗网络（GAN）；2020–2024 年间的主流方案。 |
| 平均意见得分（Mean Opinion Score, MOS） | 主观质量 | 人类评分员给出的 1–5 分平均意见得分。 |
| 说话人嵌入余弦相似度（Speaker Embedding Cosine Similarity, SECS） | 声音克隆评估指标 | 目标说话人与输出说话人嵌入向量之间的余弦相似度。 |
| F5-TTS | 2024 年开源最先进模型 | 基于流匹配（Flow-matching）扩散模型；支持零样本（Zero-shot）克隆。 |
| Kokoro | CPU 端英语 TTS 领先模型 | 8200 万参数模型，采用 Apache 2.0 许可证。 |

## 延伸阅读

- [Shen 等人 (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — 序列到序列（Seq2Seq）基线模型。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端基于流的模型。
- [Chen 等人 (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源最先进（SOTA）模型。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 截至 2026 年仍在广泛部署的声码器。
- [HuggingFace 上的 Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年面向 CPU 优化的英语 TTS 模型。