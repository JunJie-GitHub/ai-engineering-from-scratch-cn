# 音频基础 — 波形（Waveforms）、采样（Sampling）与傅里叶变换（Fourier Transform）

> 波形是原始信号。频谱图（Spectrograms）是信号的表征形式。梅尔特征（Mel features）则是机器学习友好的格式。每一条现代自动语音识别（ASR）与文本转语音（TTS）流水线都遵循这一处理阶梯，而踏上第一级台阶的关键在于理解采样与傅里叶变换。

**类型：** 学习
**编程语言：** Python
**前置知识：** 第一阶段 · 06（向量与矩阵），第一阶段 · 14（概率分布）
**预计耗时：** 约45分钟

## 核心问题

麦克风输出的是声压-时间信号，而你的神经网络（Neural Network）消费的则是张量（Tensor）。在这两者之间，存在着一系列数据处理规范。一旦违反这些规范，就会引发难以察觉的静默缺陷（Silent Bugs）：模型训练看似正常，但词错误率（Word Error Rate, WER）却翻倍；或者 TTS 系统输出的音频带有明显的嘶嘶底噪；亦或是语音克隆（Voice Cloning）系统记住了麦克风的硬件特征，而非说话人的音色。

语音系统中的每一个缺陷，最终都可以归结为以下三个问题之一：

1. 数据的录制采样率（Sample Rate）是多少？模型期望的采样率又是多少？
2. 信号是否发生了混叠（Aliasing）？
3. 你是在原始采样点上进行操作，还是在频域表征（Frequency Representation）上进行操作？

只要正确处理这些问题，第六阶段（Phase 6）的后续内容便易于掌握。一旦出错，即便是 Whisper-Large-v4 模型也会输出毫无价值的结果。

## 核心概念

![波形、采样、离散傅里叶变换 (DFT) 与频率分箱 (Frequency Bin) 可视化](../assets/audio-fundamentals.svg)

**波形 (Waveform)。** 一个取值范围为 `[-1.0, 1.0]` 的浮点数一维数组。通过采样点编号进行索引。若要转换为秒数，需除以采样率：`t = n / sr`。一段 16 kHz 采样率、时长 10 秒的音频片段，对应一个包含 160,000 个浮点数的数组。

**采样率 (Sampling Rate, sr)。** 每秒的采样点数。2026 年常用的采样率包括：

| 采样率 | 用途 |
|------|-----|
| 8 kHz | 电话通信、传统网络语音 (VoIP)。4 kHz 的奈奎斯特频率会导致辅音丢失。不建议用于自动语音识别 (Automatic Speech Recognition, ASR)。 |
| 16 kHz | 自动语音识别 (ASR) 标准。Whisper、Parakeet、SeamlessM4T v2 均接收 16 kHz 输入。 |
| 22.05 kHz | 旧版模型的文本转语音 (Text-to-Speech, TTS) 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 影视、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农采样定理 (Nyquist-Shannon Sampling Theorem)。** 采样率 `sr` 能够无歧义地表示最高至 `sr/2` 的频率。`sr/2` 这一边界即为*奈奎斯特频率 (Nyquist Frequency)*。高于奈奎斯特频率的能量会发生*混叠 (Aliasing)*——折叠至较低频段——从而破坏信号。在下采样之前，务必先进行低通滤波。

**位深度 (Bit Depth)。** 16 位脉冲编码调制 (Pulse Code Modulation, PCM)（有符号 int16，范围 ±32,767）是通用的交换格式。音乐通常使用 24 位，内部数字信号处理 (Digital Signal Processing, DSP) 则使用 32 位浮点数。`soundfile` 等库读取 int16 数据，但对外暴露的是 `[-1, 1]` 范围内的 float32 数组。

**傅里叶变换 (Fourier Transform)。** 任何有限信号均可表示为不同频率正弦波的叠加。离散傅里叶变换 (Discrete Fourier Transform, DFT) 针对 `N` 个采样点，计算出 `N` 个复数系数——每个频率分箱 (Frequency Bin) 对应一个。`bin k` 映射的频率为 `k · sr / N` Hz。模值表示该频率的幅度，角度表示相位。

**快速傅里叶变换 (Fast Fourier Transform, FFT)。** 当 `N` 为 2 的幂次时，用于计算 DFT 的 `O(N log N)` 复杂度算法。所有音频库底层均依赖 FFT。在 16 kHz 采样率下，对 1024 个采样点进行 FFT 运算，可得到 512 个有效频率分箱，覆盖 0–8 kHz 范围，频率分辨率为 15.6 Hz。

**分帧与加窗 (Framing + Windowing)。** 我们不会对整段音频直接进行 FFT。而是将其切分为相互重叠的*帧 (Frames)*（通常为 25 ms 帧长，10 ms 帧移 (Hop)），将每帧乘以一个窗函数 (Window Function)（如汉宁窗 Hann、汉明窗 Hamming）以消除边缘不连续性，随后对每一帧执行 FFT。这一过程即为短时傅里叶变换 (Short-Time Fourier Transform, STFT)。第 02 课将从此处继续展开。

## 动手实践

### 步骤 1：读取音频片段并绘制波形图

`code/main.py` 仅使用标准库 `wave` 模块，以保持演示代码无外部依赖。在生产环境中，你将使用 `soundfile` 或 `torchaudio.load`（两者均返回 `(waveform, sr)` 元组）：

import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int

### 步骤 2：从基本原理合成正弦波

import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]

在 16 kHz 采样率下，持续 1 秒的 440 Hz 正弦波（标准音 A）包含 16,000 个浮点数。使用 `wave.open(..., "wb")` 并以 16 位脉冲编码调制 (PCM) 编码进行写入。

### 步骤 3：手动计算离散傅里叶变换 (DFT)

def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out

时间复杂度为 `O(N²)` —— 对于 `N=256` 验证正确性尚可，但处理真实音频时毫无用处。实际代码应调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤 4：寻找主频 (Dominant Frequency)

幅度峰值索引 `k_star` 映射到的频率为 `k_star * sr / N`。将此算法应用于 440 Hz 正弦波时，应在频点索引 `440 * N / sr` 处返回峰值。

### 步骤 5：演示混叠 (Aliasing)

以 10 kHz 采样率对 7 kHz 正弦波进行采样（奈奎斯特频率 (Nyquist Frequency) 为 5 kHz）。7 kHz 音调高于奈奎斯特频率，会折叠至 `10 − 7 = 3 kHz`。快速傅里叶变换 (FFT) 的峰值将出现在 3 kHz 处。这是经典的混叠演示，也是每个数模/模数转换器 (DAC/ADC) 都内置砖墙式低通滤波器 (brick-wall low-pass filter) 的原因。

## 实际应用

你在 2026 年实际部署的技术栈如下：

| 任务 | 库 | 原因 |
|------|---------|-----|
| 读取/写入 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 速度最快、稳定，直接返回 float32。 |
| 重采样 (Resample) | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠 (Anti-aliasing) 处理。 |
| 短时傅里叶变换 (STFT) / 梅尔频谱 (Mel) | `torchaudio` 或 `librosa` | 对 GPU 友好；属于 PyTorch 生态系统。 |
| 实时流处理 | `sounddevice` 或 `pyaudio` | 跨平台的 PortAudio 绑定。 |
| 检查文件信息 | `ffprobe` 或 `soxi` | 命令行工具，速度快，可报告采样率/声道数/编解码器。 |

决策原则：**在处理其他任何参数之前，先确保采样率匹配**。Whisper 模型期望的输入为 16 kHz 单声道 float32 格式。若传入 44.1 kHz 立体声音频，你将得到看似模型存在缺陷的无效输出。

## 交付与部署

保存为 `outputs/skill-audio-loader.md`。该技能模块可帮助你验证音频输入是否符合下游模型的预期，并在不匹配时执行正确的重采样。

## 练习

1. **简单。** 在 16 kHz 采样率下，合成一段 1 秒长的 220 Hz + 440 Hz + 880 Hz 混合信号。运行离散傅里叶变换（DFT）。确认在预期的频点（bin）处出现三个峰值。
2. **中等。** 以 48 kHz 采样率录制一段 3 秒的 WAV 格式人声。使用 `torchaudio.transforms.Resample`（启用抗混叠）将其降采样至 16 kHz，随后再使用朴素抽取法（每三个样本保留一个）将其降至 16 kHz。对两种结果分别进行快速傅里叶变换（FFT）。混叠（aliasing）现象出现在何处？
3. **困难。** 仅使用 `math` 模块与第 3 步实现的 DFT，从零开始构建短时傅里叶变换（STFT）。设置帧长为 400，帧移为 160，使用汉宁窗（Hann window）。利用 `matplotlib.pyplot.imshow` 绘制幅度图。此即第 02 课中的语谱图（spectrogram）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 采样率（Sample rate） | 每秒的采样点数 | 模数转换器（ADC）对信号进行采样的频率（Hz）。 |
| 奈奎斯特频率（Nyquist） | 可表示的最高频率 | `sr/2`；高于该频率的能量会向下混叠。 |
| 位深度（Bit depth） | 单个样本的精度 | `int16` = 65,536 个量化级；`float32` = 在 `[-1, 1]` 区间内具备 24 位精度。 |
| 离散傅里叶变换（DFT） | 序列的傅里叶变换 | `N` 个样本 → `N` 个复数频域系数。 |
| 快速傅里叶变换（FFT） | 快速版 DFT | 时间复杂度为 `O(N log N)` 的算法，要求 `N` 为 2 的幂。 |
| 频点（Bin） | 频率列 | `k · sr / N` Hz；频率分辨率 = `sr / N`。 |
| 短时傅里叶变换（STFT） | 语谱图的底层实现 | 随时间进行的分帧与加窗 FFT。 |
| 混叠（Aliasing） | 诡异的频率“幽灵” | 高于奈奎斯特频率的能量镜像折叠至较低频点。 |

## 扩展阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理的奠基论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费且权威的数字信号处理（DSP）教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 附带代码的实战指南。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 解释现实世界音频为何并非纯净正弦波的参考资料。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟厘清频点（frequency bin）的直观理解。