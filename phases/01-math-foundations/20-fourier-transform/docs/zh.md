# 傅里叶变换 (Fourier Transform)

> 任何信号都是正弦波的叠加。傅里叶变换会告诉你具体包含哪些正弦波。

**类型：** 构建实践
**语言：** Python
**前置要求：** 第一阶段，第 01-04 课、第 19 课（复数）
**预计耗时：** 约 90 分钟

## 学习目标

- 从零实现离散傅里叶变换 (Discrete Fourier Transform, DFT)，并与时间复杂度为 O(N log N) 的库利-图基快速傅里叶变换 (Cooley-Tukey Fast Fourier Transform, FFT) 进行对比验证
- 解读频率系数：从信号中提取振幅、相位和功率谱 (Power Spectrum)
- 应用卷积定理 (Convolution Theorem)，通过 FFT 乘法执行卷积操作
- 将傅里叶频率分解与 Transformer 位置编码 (Positional Encoding) 及卷积神经网络 (Convolutional Neural Network, CNN) 的卷积层建立联系

## 问题背景

音频录音是随时间变化的气压测量值序列，股票价格是逐日波动的数值序列，图像则是空间上像素强度的二维网格。这些都属于时域 (Time Domain)（或空域 (Spatial Domain)）数据，其共同特征是数值随特定索引发生变化。

然而，许多潜在模式在时域中难以察觉。这段音频信号是单一频率的纯音还是多音叠加的和弦？该股票价格是否存在周度周期？该图像是否包含重复纹理？这些问题本质上都在探讨信号的频率成分，而时域表示恰恰掩盖了这些信息。

傅里叶变换的作用正是将数据从时域转换至频域 (Frequency Domain)。它接收一个信号，并将其分解为一系列不同频率的正弦波。每个正弦波都包含振幅（表示强度）和相位（表示起始位置），而傅里叶变换能够同时揭示这两个关键参数。

理解这一点对机器学习至关重要，因为频域思维在算法设计中无处不在。卷积神经网络的核心操作是卷积，而在频域中，卷积等价于乘法运算。Transformer 的位置编码利用频率分解来表示序列位置。音频模型（如语音识别、音乐生成）通常基于频谱图 (Spectrogram) 进行运算，这正是声音的频域表示。时间序列模型则致力于挖掘数据中的周期性模式。掌握傅里叶变换，将为你深入理解和应用上述所有技术提供必要的理论基础与专业术语。

## 核心概念

### DFT（离散傅里叶变换）的定义

给定 N 个样本 x[0], x[1], ..., x[N-1]，离散傅里叶变换（Discrete Fourier Transform, DFT）会生成 N 个频率系数 X[0], X[1], ..., X[N-1]：

X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)

for k = 0, 1, ..., N-1

每个 X[k] 都是一个复数。其模长 |X[k]| 表示频率 k 的振幅（amplitude）。其相位角 angle(X[k]) 表示该频率的相位偏移（phase offset）。

核心要点：`e^(-2*pi*i*k*n/N)` 是一个以频率 k 旋转的相量（phasor）。DFT 计算的是信号与 N 个等间隔频率之间的相关性（correlation）。如果信号在频率 k 处包含能量，相关性就会很大；否则，相关性接近于零。

### 各系数的含义

**X[0]：直流分量（DC component）。** 它是所有样本的总和，与均值成正比。它表示信号的恒定（零频）偏移量。

X[0] = sum_{n=0}^{N-1} x[n] * e^0 = sum of all samples

**1 <= k <= N/2 的 X[k]：正频率（positive frequencies）。** X[k] 表示每 N 个样本中包含 k 个周期。k 值越大，频率越高（振荡越快）。

**X[N/2]：奈奎斯特频率（Nyquist frequency）。** 使用 N 个样本所能表示的最高频率。超过此频率会发生混叠（aliasing），即高频信号伪装成低频信号。

**N/2 < k < N 的 X[k]：负频率（negative frequencies）。** 对于实值信号，满足 X[N-k] = conj(X[k])。负频率是正频率的镜像。这就是为什么有效信息仅包含在前 N/2 + 1 个系数中。

### 逆离散傅里叶变换（Inverse DFT）

逆离散傅里叶变换（Inverse DFT）从频率系数中重建原始信号：

x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(2*pi*i*k*n/N)

for n = 0, 1, ..., N-1

与正向 DFT 的唯一区别在于：指数部分的符号为正（而非负），并且包含一个 1/N 的归一化因子（normalization factor）。

逆 DFT 能够实现完美重建，不会丢失任何信息。你可以在时域（time domain）和频域（frequency domain）之间无损往返。DFT 本质上是一种基变换（change of basis）——它在不同的坐标系中重新表达了相同的信息。

### 快速傅里叶变换（FFT）：提升计算速度

如上定义的 DFT 时间复杂度为 O(N^2)：对于 N 个输出系数中的每一个，都需要对 N 个输入样本求和。当 N = 100 万时，计算量高达 10^12 次操作。

快速傅里叶变换（Fast Fourier Transform, FFT）以 O(N log N) 的复杂度计算出相同的结果。当 N = 100 万时，计算量约为 2000 万次，而非一万亿次。这正是频域分析得以实际应用的关键。

Cooley-Tukey 算法（最常见的 FFT 实现）采用分治法（divide and conquer）：

1. 将信号拆分为偶数索引和奇数索引的样本。
2. 递归计算每一半的 DFT。
3. 使用“旋转因子（twiddle factors）” `e^(-2*pi*i*k/N)` 将两个半尺寸的 DFT 结果合并。

X[k] = E[k] + e^(-2*pi*i*k/N) * O[k]          for k = 0, ..., N/2 - 1
X[k + N/2] = E[k] - e^(-2*pi*i*k/N) * O[k]    for k = 0, ..., N/2 - 1

where E = DFT of even-indexed samples
      O = DFT of odd-indexed samples

这种对称性意味着递归的每一层执行 O(N) 的工作量，共有 log2(N) 层。总计：O(N log N)。

graph TD
    subgraph "8-point FFT (Cooley-Tukey)"
        X["x[0..7]<br/>8 samples"] -->|"split even/odd"| E["Even: x[0,2,4,6]"]
        X -->|"split even/odd"| O["Odd: x[1,3,5,7]"]
        E -->|"4-pt FFT"| EK["E[0..3]"]
        O -->|"4-pt FFT"| OK["O[0..3]"]
        EK -->|"combine with twiddle factors"| XK["X[0..7]"]
        OK -->|"combine with twiddle factors"| XK
    end
    subgraph "Complexity"
        C1["DFT: O(N^2) = 64 multiplications"]
        C2["FFT: O(N log N) = 24 multiplications"]
    end

FFT 要求信号长度为 2 的幂。在实际应用中，通常会对信号进行补零（zero-padding）至下一个 2 的幂。

### 频谱分析（Spectral analysis）

**功率谱（power spectrum）** 为 |X[k]|^2，即每个频率系数模长的平方。它展示了每个频率上包含的能量大小。

**相位谱（phase spectrum）** 为 angle(X[k])，即每个频率的相位偏移。对于大多数分析任务，我们主要关注功率谱，而忽略相位。

Power at frequency k:  P[k] = |X[k]|^2 = X[k].real^2 + X[k].imag^2
Phase at frequency k:  phi[k] = atan2(X[k].imag, X[k].real)

### 频率分辨率（Frequency resolution）

DFT 的频率分辨率取决于样本数量 N 和采样率（sampling rate）fs。

Frequency of bin k:      f_k = k * fs / N
Frequency resolution:    delta_f = fs / N
Maximum frequency:       f_max = fs / 2  (Nyquist)

要区分两个相近的频率，需要更多的样本。要捕获高频信号，则需要更高的采样率。

### 卷积定理（Convolution theorem）

这是信号处理中最重要的结论之一，与卷积神经网络（CNNs）直接相关。

**时域中的卷积等于频域中的逐点乘法（pointwise multiplication）。**

x * h = IFFT(FFT(x) . FFT(h))

where * is convolution and . is element-wise multiplication

为什么这很重要：

- 对长度分别为 N 和 M 的两个信号进行直接卷积需要 O(N*M) 次操作。
- 基于 FFT 的卷积仅需 O(N log N)：分别进行变换、相乘、再逆变换。
- 对于大尺寸卷积核（kernel），FFT 卷积的速度要快得多。
- 这正是具有大感受野（receptive field）的卷积层中所发生的过程。

注意：DFT 计算的是循环卷积（circular convolution，信号会首尾相接）。对于线性卷积（linear convolution，无首尾相接），在计算前需将两个信号补零至长度 N + M - 1。

graph LR
    subgraph "Time Domain"
        TA["Signal x[n]"] -->|"convolve (slow: O(NM))"| TC["Output y[n]"]
        TB["Filter h[n]"] -->|"convolve"| TC
    end
    subgraph "Frequency Domain"
        FA["FFT(x)"] -->|"multiply (fast: O(N))"| FC["FFT(x) * FFT(h)"]
        FB["FFT(h)"] -->|"multiply"| FC
        FC -->|"IFFT"| FD["y[n]"]
    end
    TA -.->|"FFT"| FA
    TB -.->|"FFT"| FB
    FD -.->|"same result"| TC

### 加窗（Windowing）

DFT 假设信号是周期性的——它将 N 个样本视为无限重复信号的一个周期。如果信号的起始值和结束值不同，会在边界处产生不连续性，从而表现为虚假的高频成分。这种现象称为频谱泄漏（spectral leakage）。

加窗通过在计算 DFT 前将信号两端逐渐衰减至零来减少泄漏。

常用窗函数：

| 窗函数 | 形状 | 主瓣宽度 | 旁瓣电平 | 适用场景 |
|--------|-------|----------------|-----------------|----------|
| 矩形窗（Rectangular） | 平坦（无窗） | 最窄 | 最高（-13 dB） | 信号在 N 个样本内严格周期重复时 |
| 汉宁窗（Hann） | 升余弦 | 适中 | 较低（-31 dB） | 通用频谱分析 |
| 汉明窗（Hamming） | 修正余弦 | 适中 | 更低（-42 dB） | 音频处理、语音分析 |
| 布莱克曼窗（Blackman） | 三重余弦 | 较宽 | 极低（-58 dB） | 旁瓣抑制要求极高时 |

Hann window:    w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))

在 DFT 之前，将窗函数与信号逐元素相乘即可应用加窗：`X = DFT(x * w)`。

### DFT 的性质

| 性质 | 时域 | 频域 |
|----------|-------------|-----------------|
| 线性（Linearity） | a*x + b*y | a*X + b*Y |
| 时移（Time shift） | x[n - k] | X[f] * e^(-2*pi*i*f*k/N) |
| 频移（Frequency shift） | x[n] * e^(2*pi*i*f0*n/N) | X[f - f0] |
| 卷积（Convolution） | x * h | X * H（逐点相乘） |
| 乘法（Multiplication） | x * h（逐点相乘） | X * H（循环卷积，缩放 1/N） |
| 帕塞瓦尔定理（Parseval's theorem） | sum \|x[n]\|^2 | (1/N) * sum \|X[k]\|^2 |
| 共轭对称性（实数输入） | x[n] 为实数 | X[k] = conj(X[N-k]) |

帕塞瓦尔定理表明，信号在两个域中的总能量是相同的。能量在变换过程中保持守恒。

### 与位置编码（Positional encodings）的联系

原始 Transformer 模型使用正弦位置编码：

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

每一对维度 (2i, 2i+1) 以不同的频率振荡。这些频率从高（维度 0,1）到低（最后几个维度）呈几何级数分布。这使得每个位置在所有频带上都具有独特的模式——类似于傅里叶系数如何唯一标识一个信号。

它提供的关键特性包括：

- **唯一性：** 没有任何两个位置具有相同的编码。
- **值域有界：** sin 和 cos 的值始终在 [-1, 1] 之间。
- **相对位置：** 位置 p+k 的编码可以表示为位置 p 编码的线性函数。模型能够学习关注相对位置。

### 与 CNN 的联系

卷积层通过将学习到的滤波器（filter/kernel）在信号或图像上滑动来应用于输入。在数学上，这就是卷积操作。

根据卷积定理，这等价于：
1. 对输入进行 FFT
2. 对卷积核进行 FFT
3. 在频域中相乘
4. 对结果进行 IFFT（逆快速傅里叶变换）

标准的 CNN 实现通常使用直接卷积（对于小型 3x3 卷积核更快）。但对于大卷积核或全局卷积，基于 FFT 的方法要快得多。一些架构（如 FNet）完全用 FFT 替代了注意力机制（attention），以 O(N log N) 的复杂度实现了具有竞争力的精度，而非 O(N^2)。

### 频谱图与短时傅里叶变换（STFT）

单次 FFT 只能给出整个信号的频率成分，但无法告知这些频率何时出现。线性调频信号（chirp，频率随时间增加的信号）和和弦（chord，所有频率同时存在）可能具有相同的幅度谱。

短时傅里叶变换（Short-Time Fourier Transform, STFT）通过对信号的重叠窗口计算 FFT 来解决此问题。结果生成频谱图（spectrogram）：一种二维表示，一个轴为时间，另一个轴为频率。每个点的强度表示该时刻该频率的能量。

STFT procedure:
1. Choose a window size (e.g., 1024 samples)
2. Choose a hop size (e.g., 256 samples -- 75% overlap)
3. For each window position:
   a. Extract the windowed segment
   b. Apply a Hann/Hamming window
   c. Compute FFT
   d. Store the magnitude spectrum as one column of the spectrogram

频谱图是音频机器学习模型的标准输入表示。语音识别模型（如 Whisper、DeepSpeech）基于梅尔频谱图（mel-spectrograms）运行——这是一种将频率映射到梅尔刻度（mel scale）的频谱图，更符合人类对音高的感知。

### 混叠（Aliasing）

如果信号包含高于 fs/2（奈奎斯特频率）的频率，以 fs 速率采样将产生混叠副本。以 100 Hz 采样的 90 Hz 信号看起来与 10 Hz 信号完全相同。仅凭采样点无法区分它们。

Example:
  True signal: 90 Hz sine wave
  Sampling rate: 100 Hz
  Apparent frequency: 100 - 90 = 10 Hz

  The samples from the 90 Hz signal at 100 Hz sampling rate
  are identical to the samples from a 10 Hz signal.
  No amount of math can recover the original 90 Hz.

这就是为什么模数转换器（ADC）包含抗混叠滤波器（anti-aliasing filters），在采样前滤除高于奈奎斯特频率的成分。在机器学习中，如果在没有适当低通滤波的情况下对特征图进行下采样，就会出现混叠——一些架构通过抗混叠池化层（anti-aliased pooling layers）来解决此问题。

### 补零无法提高分辨率

一个常见的误解：在 FFT 前对信号补零可以提高频率分辨率。事实并非如此。补零仅在现有的频率仓（frequency bins）之间进行插值，使频谱看起来更平滑。但它无法揭示原始样本中不存在的频率细节。

真实的频率分辨率仅取决于观测时间 T = N / fs。要区分间隔为 delta_f 的两个频率，至少需要 T = 1 / delta_f 秒的数据。无论补多少零，都无法改变这一根本限制。

## 动手实现

### 步骤 1：从零实现离散傅里叶变换 (Discrete Fourier Transform)

O(N^2) 复杂度的离散傅里叶变换可直接由其数学定义推导得出。

import math

class Complex:
    ...

def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * w
        result.append(total)
    return result

### 步骤 2：逆离散傅里叶变换 (Inverse Discrete Fourier Transform)

整体结构相同，仅指数符号取正，且最终结果需除以 N。

def idft(X):
    N = len(X)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            total = total + X[k] * w
        result.append(Complex(total.real / N, total.imag / N))
    return result

### 步骤 3：快速傅里叶变换 (Fast Fourier Transform)（Cooley-Tukey 算法）

递归实现的快速傅里叶变换要求输入序列长度为 2 的幂次。算法将序列按奇偶索引拆分并递归处理，最后利用旋转因子 (twiddle factors) 进行合并。

def fft(x):
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    if N % 2 != 0:
        return dft(x)

    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result

### 步骤 4：频谱分析 (Spectral analysis) 辅助函数

def power_spectrum(X):
    return [xk.real ** 2 + xk.imag ** 2 for xk in X]

def convolve_fft(x, h):
    N = len(x) + len(h) - 1
    padded_N = 1
    while padded_N < N:
        padded_N *= 2

    x_padded = x + [0.0] * (padded_N - len(x))
    h_padded = h + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    Y = [xk * hk for xk, hk in zip(X, H)]

    y = idft(Y)
    return [y[n].real for n in range(N)]

## 实际应用

在实际工作中，请使用 `numpy` 的快速傅里叶变换 (FFT)，它由高度优化的 C 语言库提供支持。

import numpy as np

signal = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(256, d=1/256)

power = np.abs(spectrum) ** 2

positive_freqs = freqs[:len(freqs)//2]
positive_power = power[:len(power)//2]

对于加窗处理 (windowing) 和更高级的频谱分析 (spectral analysis)：

from scipy.signal import windows, stft

window = windows.hann(256)
windowed = signal * window
spectrum = np.fft.fft(windowed)

对于卷积 (convolution)：

from scipy.signal import fftconvolve

result = fftconvolve(signal, kernel, mode='full')

对于频谱图 (spectrogram)：

from scipy.signal import stft

frequencies, times, Zxx = stft(signal, fs=sample_rate, nperseg=256)
spectrogram = np.abs(Zxx) ** 2

频谱图矩阵的形状为 `(n_frequencies, n_time_frames)`。每一列代表一个时间窗口内的功率谱 (power spectrum)。这正是音频机器学习 (ML) 模型所使用的输入数据。

## 发布

运行 `code/fourier.py` 以生成 `outputs/prompt-spectral-analyzer.md`。

## 练习

1. **纯音识别。** 创建一个包含单一正弦波的信号，其频率未知（介于 1 到 50 Hz 之间），采样率为 128 Hz，持续 1 秒。使用你的离散傅里叶变换 (DFT) 来识别该频率。验证结果是否匹配。现在添加标准差为 0.5 的高斯噪声 (Gaussian noise) 并重复实验。噪声会对频谱产生什么影响？

2. **FFT 与 DFT 验证。** 生成长度为 64 的随机信号。分别计算 DFT（时间复杂度为 O(N^2)）和 FFT。验证所有系数的匹配误差是否在 1e-10 以内。对长度分别为 256、512、1024 和 2048 的信号，分别测量这两个函数的运行时间。绘制 DFT 耗时与 FFT 耗时的比值图。

3. **通过实例验证卷积定理。** 创建信号 `x = [1, 2, 3, 4, 0, 0, 0, 0]` 和滤波器 `h = [1, 1, 1, 0, 0, 0, 0, 0]`。直接计算它们的循环卷积 (circular convolution)（使用嵌套循环）。然后通过 FFT 计算（变换、相乘、逆变换）。验证结果是否一致。现在通过适当的零填充 (zero-padding) 进行线性卷积 (linear convolution)。

4. **加窗效应。** 创建一个由 10 Hz 和 12 Hz（频率非常接近）两个正弦波叠加而成的信号。以 128 Hz 采样 1 秒。分别在不加窗、使用汉宁窗 (Hann window) 和汉明窗 (Hamming window) 的情况下计算功率谱。哪种窗口最容易区分这两个峰值？为什么？

5. **位置编码分析。** 为 `d_model = 128` 和 `max_pos = 512` 生成正弦位置编码 (sinusoidal positional encodings)。对于每一对位置 `(p1, p2)`，计算它们编码的点积 (dot product)。证明该点积仅取决于 `|p1 - p2|`，而与绝对位置无关。随着距离增加，点积会发生什么变化？

## 关键术语

| 术语 | 含义 |
|------|---------------|
| DFT（离散傅里叶变换，Discrete Fourier Transform） | 将 N 个时域采样点转换为 N 个频域系数。每个系数代表信号与对应频率复正弦波的相关性 |
| FFT（快速傅里叶变换，Fast Fourier Transform） | 一种时间复杂度为 O(N log N) 的 DFT 计算算法。Cooley-Tukey 算法通过递归方式将偶数/奇数索引进行拆分 |
| 逆离散傅里叶变换（Inverse DFT） | 根据频域系数重建时域信号。其公式与 DFT 相同，仅指数符号相反且需乘以 1/N 进行缩放 |
| 频率仓（Frequency bin） | DFT 输出中的每个索引 k 代表频率 k*fs/N Hz。“仓”（bin）即离散的频率区间 |
| 直流分量（DC component） | X[0]，即零频系数。其值与信号均值成正比 |
| 奈奎斯特频率（Nyquist frequency） | fs/2，在采样率 fs 下可表示的最高频率。高于此频率的信号会发生混叠 |
| 功率谱（Power spectrum） | \|X[k]\|^2，即各频率系数幅值的平方。用于展示能量在不同频率上的分布情况 |
| 相位谱（Phase spectrum） | angle(X[k])，即各频率分量的相位偏移。在分析中常被忽略 |
| 频谱泄漏（Spectral leakage） | 将非周期信号视为周期信号处理时产生的虚假频率成分。可通过加窗操作来抑制 |
| 窗函数（Window function） | 在 DFT 前应用的渐缩函数（如 Hann、Hamming、Blackman），用于减少频谱泄漏 |
| 旋转因子（Twiddle factor） | 复指数 e^(-2*pi*i*k/N)，在 FFT 蝶形运算中用于组合子 DFT |
| 卷积定理（Convolution theorem） | 时域中的卷积等价于频域中的逐点相乘。该定理是信号处理与卷积神经网络（CNN）的基础 |
| 循环卷积（Circular convolution） | 信号首尾相接进行的卷积运算。DFT 天然计算的就是循环卷积 |
| 线性卷积（Linear convolution） | 无首尾相接的标准卷积运算。通过在 DFT 前进行零填充（zero-padding）来实现 |
| 帕塞瓦尔定理（Parseval's theorem） | 傅里叶变换前后信号的总能量保持不变。即 sum \|x[n]\|^2 = (1/N) sum \|X[k]\|^2 |
| 混叠（Aliasing） | 因采样率不足，导致高于奈奎斯特频率的成分在频谱中表现为低频信号的现象 |

## 延伸阅读

- [Cooley & Tukey：用于计算机计算复傅里叶级数的算法 (1965)](https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/) - 改变计算领域的快速傅里叶变换（Fast Fourier Transform, FFT）原始论文
- [3Blue1Brown：傅里叶变换究竟是什么？](https://www.youtube.com/watch?v=spUNpyF58BY) - 傅里叶变换（Fourier Transform）最优质的可视化入门教程
- [Lee-Thorp 等人：FNet：使用傅里叶变换混合 Token (2021)](https://arxiv.org/abs/2105.03824) - 在 Transformer 架构中使用 FFT 替代自注意力机制（Self-Attention）
- [Smith：科学家与工程师的数字信号处理指南](http://www.dspguide.com/) - 免费在线教材，深入讲解 FFT、加窗处理（Windowing）与频谱分析（Spectral Analysis）
- [Vaswani 等人：Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - 基于傅里叶频率分解推导出的正弦位置编码（Sinusoidal Positional Encodings）
- [Radford 等人：Whisper (2022)](https://arxiv.org/abs/2212.04356) - 使用梅尔频谱图（Mel-Spectrograms）作为输入表示的语音识别模型