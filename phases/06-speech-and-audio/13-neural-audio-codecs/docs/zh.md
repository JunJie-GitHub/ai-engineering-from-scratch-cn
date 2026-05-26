# 神经音频编解码器（Neural Audio Codecs）—— EnCodec、SNAC、Mimi、DAC 与语义-声学分离（Semantic-Acoustic Split）

> 2026 年的音频生成几乎完全基于词元（tokens）。EnCodec、SNAC、Mimi 和 DAC 将连续的波形转换为离散序列，供 Transformer 进行预测。语义与声学词元的分离——首个码本（codebook）用于语义，其余用于声学——是自 Transformer 应用于音频领域以来最重要的架构演进。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 6 阶段 · 02（频谱图/Spectrograms），第 10 阶段 · 11（量化/Quantization），第 5 阶段 · 19（子词分词/Subword Tokenization）
**Time:** 约 60 分钟

## 核心问题

语言模型基于离散词元（tokens）运行，而音频是连续的。如果你想构建类似大语言模型（LLM）的语音/音乐生成模型——如 MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——你首先需要一种**神经音频编解码器（neural audio codec）**：这是一种经过训练的编码器，能将音频离散化为少量词元构成的词表，并配备一个匹配的解码器来重建波形。

目前主要衍生出两大流派：

1. **重建优先型编解码器（Reconstruction-first codecs）**—— EnCodec、DAC。以优化感知音频质量为目标。其词元属于“声学”性质——它们捕获了所有信息，包括说话人身份、音色和背景噪声。
2. **语义优先型编解码器（Semantic-first codecs）**—— Mimi（Kyutai）、SpeechTokenizer。强制首个码本（codebook）编码语言学/语音学内容（通常通过从 WavLM 进行知识蒸馏/distilling 实现）。后续码本则负责补充声学细节。

2024-2026 年的关键洞察：**纯重建型编解码器在尝试从文本生成语音时，会导致语音模糊不清。** 基于编解码器词元的大语言模型必须在同一个码本中同时学习语言结构和声学结构，这无法有效扩展（scale）。将二者分离——语义码本 0，声学码本 1-N——正是 Moshi 和 Sesame CSM 能够成功运行的核心原因。

## 核心概念

![四大编解码器全景：EnCodec、DAC、SNAC（多尺度）、Mimi（语义+声学）](../assets/codec-comparison.svg)

### 核心技巧：残差矢量量化（Residual Vector Quantization, RVQ）

现代音频编解码器（Audio Codec）不再依赖单一的大型码本（Codebook）（为保证音质，这通常需要数百万个码字），而是普遍采用**残差矢量量化（RVQ）**：一种由多个小型码本级联而成的结构。第一个码本对编码器输出进行量化；第二个码本对残差进行量化；依此类推。每个码本包含 1024 个码字。8 个码本组合后的有效词表大小可达 1024^8 ≈ 10^24。

在推理阶段，解码器会将每一帧所选的所有码字相加，以重建音频信号。

### 2026 年值得关注的四大编解码器

**EnCodec（Meta，2022）**。基准模型。基于波形的编码器-解码器架构，以 RVQ 作为瓶颈层。支持 24 kHz 采样率，最多可使用 32 个码本，默认配置为 4 个码本 @ 1.5 kbps。采用 `1D conv + transformer + 1D conv` 架构。被 MusicGen 所采用。

**DAC（Descript，2023）**。采用 L2 归一化码本、周期性激活函数及优化损失函数的 RVQ 架构。在所有开源编解码器中拥有最高的重建保真度——在使用 12 个码本时，有时甚至无法与原始语音区分。支持 44.1 kHz 全频段。

**SNAC（Hubert Siuzdak，2024）**。多尺度 RVQ——粗粒度码本的帧率低于细粒度码本。以分层方式对音频进行建模：约 12 Hz 的粗粒度“草图”加上 50 Hz 的细节。被 Orpheus-3B 采用，因为其分层结构能很好地映射到基于语言模型（Language Model, LM）的生成流程中。

**Mimi（Kyutai，2024）**。2026 年的颠覆性技术。帧率仅为 12.5 Hz（极低），8 个码本 @ 4.4 kbps。码本 0 **蒸馏自 WavLM**——经过训练以预测 WavLM 的语音内容特征。码本 1-7 为声学残差。这种分离设计为 Moshi（第 15 课）和 Sesame CSM 提供了核心支持。

### 帧率对语言模型至关重要

更低的帧率 = 更短的序列 = 更快的语言模型推理速度。

| 编解码器 | 帧率 | 1 秒 = N 帧 | 适用场景 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k（粗粒度） | ~12 Hz | 12 | 高效自回归语言模型（AR-LM） |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

在 12.5 Hz 的帧率下，一段 10 秒的语音仅包含 125 个编解码器帧——Transformer 模型可以轻松完成预测。

### 语义 Token 与声学 Token

frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]

- **语义 Token（Mimi 中的码本 0）**。编码“说了什么”——音素、词汇、内容。通过辅助预测损失从 WavLM 蒸馏而来。
- **声学 Token（码本 1-7）**。编码音色、说话人身份、韵律、背景噪声及精细细节。

自回归语言模型（Autoregressive Language Model, AR LM）会首先预测语义 Token（以文本为条件），随后预测声学 Token（以语义 Token 和说话人参考音频为条件）。这种分解策略正是现代语音合成（Text-to-Speech, TTS）能够实现零样本声音克隆的原因：语义模型负责处理内容，声学模型负责处理音色。

### 2026 年重建质量（比特率越低越好）

| 编解码器 | 比特率 | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

像 Opus 这样的传统编解码器在单位比特率的感知质量上依然占优。而神经编解码器（Neural Codec）的优势在于**离散 Token（Discrete Token）**（Opus 无法生成此类输出）以及**生成模型质量**（语言模型能够利用这些 Token 实现的能力）。

## 构建它

### 步骤 1：使用 EnCodec 进行编码 (encode)

from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64

在 6 kbps 码率下，`n_codebooks=8`。每个码本索引 (code) 的取值范围为 0-1023（10 位）。

### 步骤 2：解码 (decode) 并评估重建质量 (reconstruction)

with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()

### 步骤 3：语义-声学分离 (semantic-acoustic split)（Mimi 风格）

from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]

语义码本 0 与 WavLM 对齐。你可以训练一个文本到语义的 Transformer (Transformer)——其词表 (vocabulary) 远小于直接生成音频的模型。随后，一个独立的声学到波形解码器 (acoustic-to-waveform decoder) 将以说话人参考音频为条件 (conditions on a speaker reference) 进行生成。

### 步骤 4：为什么基于编解码器 Token (codec tokens) 的自回归语言模型 (Autoregressive Language Model, AR LM) 有效

对于一段 10 秒的语音片段，在 Mimi 的 12.5 Hz × 8 个码本配置下：

N_tokens = 10 * 12.5 * 8 = 1000 tokens

1000 个 Token 对于 Transformer 来说上下文长度 (context) 微不足道。在现代 GPU 上，一个 2.56 亿参数的 Transformer 仅需几毫秒即可生成 10 秒的语音。

## 使用指南

任务与编解码器映射：

| 任务 | 编解码器 |
|------|-------|
| 通用音乐生成 | EnCodec-24k |
| 最高保真度重建 | DAC-44.1k |
| 语音自回归语言模型（TTS） | SNAC 或 Mimi |
| 流式全双工语音 | Mimi (12.5 Hz) |
| 带文本提示的音效库 | EnCodec + T5 条件控制 |
| 细粒度音频编辑 | DAC + 音频修复 (inpainting) |

经验法则：**如果你正在构建生成式模型 (generative model)，请从 Mimi 或 SNAC 开始。如果你正在构建压缩流水线 (compression pipeline)，请使用 Opus。**

## 常见陷阱

- **码本 (codebooks) 数量过多。** 增加码本会线性提升保真度 (fidelity)，但也会线性增加语言模型 (LM) 的序列长度 (sequence length)。建议控制在 8-12 个。
- **帧率 (frame-rate) 不匹配。** 在 12.5 Hz 的 Mimi 上训练语言模型，随后在 50 Hz 的 EnCodec 上进行微调 (fine-tuning)，会导致静默失败 (fails silently)。
- **假设所有码本同等重要。** 在 Mimi 中，码本 0 承载语义内容；丢失它会彻底破坏语音可懂度 (intelligibility)。而丢失码本 7 几乎察觉不到差异。
- **仅以重建质量作为评估指标。** 一个编解码器可能拥有极佳的重建效果，但如果其语义结构不佳，对于基于语言模型的生成任务将毫无用处。

## 交付

将文件保存为 `outputs/skill-codec-picker.md`。为指定的生成或压缩任务选择合适的编解码器。

## 练习

1. **简单。** 运行 `code/main.py`。它实现了一个简易的标量量化器（scalar quantizer）与残差量化器（residual quantizer）组合，并会在你逐步增加码本（codebook）数量时测量重建误差（reconstruction error）。
2. **中等。** 安装 `encodec`，并在一个独立测试集（held-out）语音片段上对比使用 1、4、8、32 个码本的效果。绘制语音质量感知评估（PESQ）或均方误差（MSE）随比特率（bitrate）变化的曲线图。
3. **困难。** 加载 Mimi 模型。对一段音频进行编码。将码本 0 替换为随机整数后进行解码。随后以同样方式替换码本 7。对比这两种扰动（corruption）带来的影响——码本 0 的扰动应会严重破坏语音可懂度（intelligibility）；而码本 7 的扰动则几乎不会引起任何明显变化。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| RVQ | 残差量化（Residual Quantization） | 小型码本的级联结构；每个码本负责对前一级的残差进行量化。 |
| 帧率（Frame Rate） | 编解码器速度 | 每秒生成的令牌帧（token-frame）数量。数值越低，语言模型（LM）推理速度越快。 |
| 语义码本（Semantic Codebook） | 码本 0（Mimi） | 从自监督学习（SSL）特征中蒸馏得到的码本；负责编码语义内容。 |
| 声学码本（Acoustic Codebooks） | 其余所有码本 | 负责编码音色、韵律、噪声及精细声学细节。 |
| PESQ / ViSQOL | 感知质量 | 与平均意见得分（MOS）高度相关的客观评估指标。 |
| EnCodec | Meta 编解码器 | RVQ 的基线模型；被 MusicGen 采用。 |
| Mimi | Kyutai 编解码器 | 12.5 Hz 帧率；采用语义-声学分离架构；为 Moshi 提供底层支持。 |

## 延伸阅读

- [Défossez 等人 (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ 基线模型。
- [Kumar 等人 (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 保真度最高的开源方案。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — 多尺度 RVQ 架构。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — 语义-声学分离架构，基于 WavLM 蒸馏。
- [Borsos 等人 (2023). AudioLM](https://arxiv.org/abs/2209.03143) — 两阶段语义/声学范式。
- [Zeghidour 等人 (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 首个支持流式处理的 RVQ 编解码器。