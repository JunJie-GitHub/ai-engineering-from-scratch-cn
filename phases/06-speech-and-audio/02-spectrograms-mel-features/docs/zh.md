# 语谱图（Spectrograms）、梅尔刻度（Mel Scale）与音频特征（Audio Features）

> 神经网络（Neural Networks）并不擅长直接处理原始波形（Raw Waveforms）。它们处理的是语谱图（Spectrograms）。而处理梅尔语谱图（Mel Spectrograms）的效果则更佳。在 2026 年，每一款自动语音识别（ASR）、文本转语音（TTS）以及音频分类器（Audio Classifier）的成败，都取决于这一项预处理（Preprocessing）选择。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 6 阶段 · 01（音频基础）
**Time:** 约 45 分钟

## 问题所在

以一段 10 秒、采样率为 16 kHz 的音频片段为例。它包含 160,000 个浮点数，取值均在 `[-1, 1]` 之间，与“狗叫”或“单词 cat”等标签几乎完全不相关。原始波形虽然包含信息，但其表现形式使得模型难以直接提取。即使两个完全相同的音素（Phonemes）相隔 100 ms 发出，它们的原始采样值也会截然不同。

语谱图（Spectrogram）正是为了解决这一问题而生。它压缩了人类听觉感知所忽略的时间细节（如微秒级的抖动），同时保留了感知所关注的结构特征（即在约 10–25 ms 的时间窗口内，哪些频率具有较高的能量）。

梅尔语谱图（Mel Spectrogram）则更进一步。人类对音高（Pitch）的感知是对数级的：100 Hz 与 200 Hz 之间的听感差异，与 1000 Hz 与 2000 Hz 之间的差异“听起来是等距的”。梅尔刻度（Mel Scale）正是通过扭曲频率轴来匹配这一特性。从 2010 年到 2026 年，经过梅尔刻度缩放的语谱图一直是语音机器学习（Speech ML）中最为关键的特征。

## 核心概念

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**短时傅里叶变换（Short-Time Fourier Transform, STFT）。** 将波形切分为重叠的帧（典型设置：25 ms 窗长，10 ms 步长 = 16 kHz 采样率下的 400 个样本 / 160 个样本）。将每一帧乘以窗函数（window function）（默认使用汉宁窗（Hann）；汉明窗（Hamming）的权衡略有不同）。对每一帧执行快速傅里叶变换（Fast Fourier Transform, FFT）。将幅度谱堆叠为形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是你的语谱图（spectrogram）。

**对数幅度（Log-magnitude）。** 原始幅度跨越 5 到 6 个数量级。使用 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围（dynamic range）。所有生产级流水线均采用对数幅度，而非原始幅度。

**梅尔刻度（Mel scale）。** 频率 `f`（单位：Hz）通过公式 `m = 2595 * log10(1 + f / 700)` 映射为梅尔值 `m`。该映射在 1 kHz 以下大致呈线性，在 1 kHz 以上大致呈对数关系。覆盖 0–8 kHz 的 80 个梅尔频带（mel bins）是自动语音识别（Automatic Speech Recognition, ASR）的标准输入。

**梅尔滤波器组（Mel filterbank）。** 一组在梅尔刻度上等间距分布的三角形滤波器。每个滤波器是相邻 FFT 频点的加权和。将 STFT 幅度与滤波器组矩阵相乘，只需一次矩阵乘法（matrix multiplication, matmul）即可得到梅尔语谱图（mel spectrogram）。

**对数梅尔语谱图（Log-mel spectrogram）。** `` `log(mel_spec + 1e-10)` ``。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年通用的音频前端（audio frontend）。

**梅尔频率倒谱系数（Mel-Frequency Cepstral Coefficients, MFCCs）。** 对对数梅尔语谱图应用离散余弦变换（Discrete Cosine Transform, DCT，II 型），并保留前 13 个系数。该操作对特征进行去相关处理并进一步压缩。在 2015 年左右之前，它一直是主导特征，直到基于原始对数梅尔谱的卷积神经网络（Convolutional Neural Networks, CNNs）与 Transformer 模型迎头赶上。目前仍用于说话人识别（speaker recognition）任务（如 x-vectors、ECAPA 模型）。

**分辨率权衡（Resolution trade-off）。** FFT 点数越大 = 频率分辨率（frequency resolution）越高，但时间分辨率（time resolution）越低。25 ms / 10 ms 是音频机器学习（audio-ML）的默认设置；音乐处理常用 50 ms / 12.5 ms；瞬态检测（transient detection）（如鼓点、爆破音（plosives））则使用 5 ms / 2 ms。

## 动手构建

### 步骤 1：波形分帧

def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]

一段 10 秒、采样率为 16 kHz 的音频片段，在设置 `frame_len=400, hop=160` 时，将生成 998 个帧 (frame)。

### 步骤 2：汉宁窗 (Hann window)

import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]

在快速傅里叶变换 (FFT) 之前进行逐元素相乘。此举可消除因在非零端点处截断信号而引发的频谱泄漏 (spectral leakage)。

### 步骤 3：短时傅里叶变换幅度 (STFT magnitude)

def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]

实际生产环境中通常使用 `torch.stft` 或 `librosa.stft`（基于 FFT 且经过向量化优化）。此处的循环仅用于教学演示；它可在 `code/main.py` 中针对较短的音频片段运行。

### 步骤 4：梅尔滤波器组 (mel filterbank)

def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb

使用 `n_fft=400` 覆盖 0–8 kHz 范围的 80 个梅尔频带，将生成一个 `(80, 201)` 的矩阵。将 `(n_frames, 201)` 的 STFT 幅度矩阵与该滤波器组的转置相乘，即可得到 `(n_frames, 80)` 的梅尔频谱图 (mel spectrogram)。

### 步骤 5：对数梅尔频谱 (log-mel)

def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]

常见的替代方案包括：`librosa.power_to_db`（基于参考值归一化的分贝值）或 `10 * log10(power + eps)`。Whisper 模型采用了更为复杂的截断与归一化流程（详见 Whisper 的 `log_mel_spectrogram` 实现）。

### 步骤 6：梅尔频率倒谱系数 (MFCCs)

def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]

对每个对数梅尔帧应用离散余弦变换 (DCT)，并保留前 13 个系数。由此即可得到你的 MFCC 矩阵。通常第一个系数会被舍弃，因为它主要编码的是整体能量信息。

## 使用方法

2026 年技术栈：

| 任务 | 特征参数 |
|------|----------|
| 自动语音识别（ASR）(Whisper, Parakeet, SeamlessM4T) | 80 个对数梅尔频谱（log-mels），10 ms 跳步（hop），25 ms 窗长（window） |
| 语音合成（TTS）声学模型 (VITS, F5-TTS, Kokoro) | 80 个梅尔频谱（mels），5–12 ms 跳步以实现精细的时间控制 |
| 音频分类 (AST, PANNs, BEATs) | 128 个对数梅尔频谱，10 ms 跳步 |
| 说话人嵌入（Speaker embedding）(ECAPA-TDNN, WavLM) | 80 个对数梅尔频谱或原始波形自监督学习（SSL）特征 |
| 音乐生成 (MusicGen, Stable Audio 2) | EnCodec 离散标记（discrete tokens）（非梅尔频谱） |
| 关键词检测（Keyword spotting） | 40 个梅尔频率倒谱系数（MFCCs），适用于微型设备 |

经验法则：**如果你不从事音乐相关任务，请从 80 个对数梅尔频谱（log-mels）开始。** 任何偏离此标准的做法都需要提供充分的理由。

## 2026 年依然常见的陷阱

- **梅尔频谱数量不匹配。** 训练时使用 80 个梅尔频谱，推理时却使用 128 个。这会导致静默失败（silent failure）。务必在输入和输出端记录特征形状（feature shape）。
- **上游采样率不匹配。** 基于 22.05 kHz 计算的梅尔频谱与 16 kHz 的结果截然不同。请在特征提取（featurization）*之前* 修正采样率（sample rate）。
- **分贝（dB）与对数（log）混淆。** Whisper 期望输入的是对数梅尔频谱（log-mel），而非分贝梅尔频谱（dB-mel）。部分 Hugging Face（HF）流水线能自动检测，但你的自定义代码不会。
- **归一化漂移（normalization drift）。** 训练时采用逐语句归一化（per-utterance normalization），推理时却使用全局归一化（global normalization）。这是一个会导致词错误率（WER）翻倍的线上缺陷。
- **填充泄漏（padding leakage）。** 在音频片段末尾进行零填充（zero-padding）会导致尾部帧的频谱变得平坦。请采用对称填充或复制填充。

## 交付上线

保存为 `outputs/skill-feature-extractor.md`。该技能模块会根据目标模型自动选择特征类型、梅尔频谱数量、帧长/跳步（frame/hop）以及归一化方式。

## 练习

1. **简单。** 运行 `code/main.py`。该脚本会合成一个线性调频信号（chirp，频率从 200 Hz 扫描至 4000 Hz），并打印每一帧中幅值最大的梅尔频带索引（argmax mel bin）。绘制图表（可选）并确认其是否与频率扫描轨迹一致。
2. **中等。** 将 `n_mels` 设置为 `{40, 80, 128}`，`frame_len` 设置为 `{200, 400, 800}` 重新运行。测量时间轴上的尖锐峰值带宽。哪种参数组合对调频信号的解析效果最佳？
3. **困难。** 实现 `power_to_db` 函数，并在 AudioMNIST 数据集上对比微型卷积神经网络（CNN）分类器的自动语音识别（ASR）准确率。测试条件包括：(a) 原始对数梅尔频谱，(b) 使用 `ref=max` 的分贝梅尔频谱，(c) 13 维梅尔频率倒谱系数（MFCC-13）+ 一阶差分（delta）+ 二阶差分（delta-delta）。报告 Top-1 准确率（top-1 accuracy）。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 帧（Frame） | 切片 | 送入单次快速傅里叶变换（FFT）的 25 ms 波形片段。 |
| 跳步（Hop） | 步长 | 相邻帧之间的采样点数；10 ms 是 ASR 的默认值。 |
| 窗函数（Window） | 汉宁/汉明窗之类的东西 | 逐点相乘的权重函数，用于将帧边缘平滑衰减至零。 |
| 短时傅里叶变换（STFT） | 频谱图生成器 | 加窗分帧后的 FFT；输出时间 × 频率矩阵。 |
| 梅尔（Mel） | 扭曲的频率 | 对数感知尺度；计算公式为 `m = 2595·log10(1 + f/700)`。 |
| 滤波器组（Filterbank） | 那个矩阵 | 将 STFT 投影到梅尔频带上的三角形滤波器。 |
| 对数梅尔（Log-mel） | Whisper 的输入 | `log(mel_spec + eps)`；2026 年已成为标准。 |
| 梅尔频率倒谱系数（MFCC） | 老派特征 | 对数梅尔频谱的离散余弦变换（DCT）；通常取 13 个系数，已去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — 梅尔频率倒谱系数（MFCC）的原始论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 梅尔刻度（Mel Scale）的原始文献。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 官方参考实现代码。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 以及帧移（hop）/窗口大小（window）的参考文档。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — 面向 Parakeet 与 Canary 模型的生产级音频预处理流水线（pipeline）。