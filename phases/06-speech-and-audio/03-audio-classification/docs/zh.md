# 音频分类（Audio Classification）—— 从基于梅尔频率倒谱系数（MFCCs）的 k近邻（k-NN）到 AST 与 BEATs

> 无论是区分“狗叫与警报声”，还是识别“这是哪种语言”，都属于音频分类的范畴。其核心特征是梅尔频谱（Mel Spectrograms）。网络架构每十年都在迭代演进，但评估指标始终围绕 AUC、F1 分数以及各类别召回率（Per-class Recall）。

**类型：** 构建（Build）
**编程语言：** Python
**前置知识：** 第 6 阶段 · 02（频谱图与梅尔频谱）、第 3 阶段 · 06（卷积神经网络 CNNs）、第 5 阶段 · 08（用于文本的 CNN 与 RNN）
**预计耗时：** 约 75 分钟

## 问题定义

给定一段 10 秒的音频片段，你的任务是判断：“它是什么？”无论是城市环境音（警报声、电钻声、狗叫）、语音指令（是/否/停止）、语种识别（英语/西班牙语/阿拉伯语）、说话人情绪（愤怒/平静），还是环境音（室内/室外、嘈杂人声），这些任务都属于*音频分类*。到了 2026 年，其基线架构已经非常成熟：对数梅尔频谱（Log-Mel）→ 卷积神经网络（CNN）或 Transformer → Softmax 层。

核心难点并不在于网络模型本身，而在于数据。音频数据集通常面临严重的类别不平衡（Class Imbalance）、强烈的域偏移（Domain Shift，如干净音频与噪声音频的差异）以及标签噪声（Label Noise，究竟是谁来界定“城市嘈杂声”与“餐厅背景音”的区别？）。该问题 80% 的工作量集中在数据整理（Data Curation）、数据增强（Data Augmentation）与模型评估上，而非简单地将 CNN 替换为 Transformer。

## 核心概念

![音频分类阶梯：从基于梅尔频率倒谱系数（MFCC）的 k-近邻算法（k-NN）到音频频谱图 Transformer（AST）再到 BEATs](../assets/audio-classification.svg)

**基于 MFCC 的 k-NN（1990 年代基线）。** 将每个音频片段的梅尔频率倒谱系数展平，计算其与已标注样本库的余弦相似度（cosine similarity），并返回前 K 个最近邻的多数投票结果。在干净的小型数据集（如 Speech Commands、ESC-50）上表现相当出色。无需 GPU 即可运行。

**基于对数梅尔频谱（log-mels）的二维卷积神经网络（2D CNN，2015-2019）。** 将 `(T, n_mels)` 的对数梅尔频谱视为图像。应用 ResNet-18 或 VGG 风格的架构。在时间轴上进行全局平均池化（global mean pooling）。对类别应用 Softmax 函数。这仍然是 2026 年大多数 Kaggle 竞赛的基线模型。

**音频频谱图 Transformer（AST，2021-2024）。** 将对数梅尔频谱划分为图像块（patchify，例如 16×16 的块），添加位置编码（position embeddings），随后输入至视觉 Transformer（ViT）。在监督学习（supervised learning）场景下，该模型在 AudioSet 数据集上达到了当时的最先进水平（state of the art，mAP 为 0.485）。

**BEATs 与 WavLM-base（2024-2026）。** 在数百万小时的音频上进行自监督预训练（self-supervised pretraining）。仅需使用原本所需监督数据量的 1% 到 10%，即可在你的任务上进行微调（fine-tune）。到 2026 年，这已成为非语音音频任务的默认起点。BEATs-iter3 在 AudioSet 上的平均精度均值（mAP）比 AST 高出 1-2 个点，同时计算量仅为后者的四分之一。

**将 Whisper 编码器作为冻结骨干网络（frozen backbone，2024）。** 提取 Whisper 的编码器，移除解码器，并接上一个线性分类器（linear classifier）。在零音频数据增强（audio augmentation）的情况下，于语言识别（language ID）和简单事件分类任务上接近最先进水平（SOTA）。堪称“免费午餐”基线。

### 类别不平衡（class imbalance）才是真正的挑战

ESC-50：50 个类别，每类 40 个片段——类别平衡，较为简单。UrbanSound8K：10 个类别，不平衡比例达 10:1。AudioSet：632 个类别，长尾（long tail）比例高达 100,000:1。行之有效的技术包括：

- 训练期间使用平衡采样（balanced sampling）（评估阶段不使用）。
- Mixup：对两个音频片段（及其标签）进行线性插值作为数据增强。
- SpecAugment：随机掩蔽（mask）时间和频率波段。方法简单，但至关重要。

### 评估（Evaluation）

- 多分类互斥任务（如 Speech Commands）：Top-1 准确率、Top-5 准确率。
- 多分类多标签任务（如 AudioSet、UrbanSound 风格）：平均精度均值（mean average precision, mAP）。
- 严重不平衡任务：各类别召回率（per-class recall）+ 宏平均 F1 分数（macro F1）。

2026 年你需要了解的关键数据：

| 基准数据集 | 基线模型 | 2026 年最先进水平（SOTA） | 来源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs 论文 (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR 排行榜 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 结果 |

## 动手构建

### 步骤 1：特征提取 (featurize)

def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]

### 步骤 2：固定长度特征汇总 (fixed-length summary)

def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var

简单却强大：沿时间轴计算均值与方差，即可为 13 系数的梅尔频率倒谱系数 (MFCC) 生成一个 26 维的固定嵌入向量 (embedding)。该过程可瞬间完成。截至 2017 年，该方法在 ESC-50 数据集上仍能击败当时最先进的神经网络 (NN) 基线模型。

### 步骤 3：K近邻算法 (k-NN)

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]

### 步骤 4：在对数梅尔频谱 (log-mels) 上升级至卷积神经网络 (CNN)

在 PyTorch 中：

import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))

模型参数量约为 300 万。在单张 RTX 4090 显卡上，于 ESC-50 数据集训练仅需约 10 分钟，准确率可达 80% 以上。

### 步骤 5：2026 年默认方案——微调 (fine-tune) BEATs 模型

from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits

若需使用 BEATs 模型，可通过 `beats` 库加载 `microsoft/BEATs-base`；其 `transformers` 应用程序接口 (API) 的调用结构与此相同。

## 使用方法

2026年技术栈：

| 场景 | 推荐起步方案 |
|-----------|-----------|
| 极小数据集（<1000个音频片段） | 基于梅尔频率倒谱系数 (MFCC) 均值的 k近邻算法 (k-NN)（你的基线模型）+ 音频增强 (Audio Augmentation) |
| 中等数据集（1K–100K） | 微调 (Fine-tune) BEATs 或 音频频谱图变换器 (AST) |
| 大型数据集（>100K） | 从头训练或微调 Whisper-encoder |
| 实时、边缘计算 | 40维 MFCC 卷积神经网络 (CNN)，量化至 int8（关键词唤醒 (KWS) 风格） |
| 多标签（AudioSet） | BEATs-iter3 配合 二元交叉熵损失 (BCE Loss) + 混合增强 (Mixup) + 频谱增强 (SpecAugment) |
| 语种识别 (Language ID) | MMS-LID，SpeechBrain VoxLingua107 基线 |

决策原则：**从冻结的骨干网络 (Frozen Backbone) 开始，而非从头构建新模型**。微调 BEATs 的分类头 (Head) 只需数小时而非数周，即可达到当前最优 (SOTA) 95% 的性能。

## 交付上线

保存为 `outputs/skill-classifier-designer.md`。针对给定的音频分类任务，选定模型架构、数据增强策略、类别平衡策略以及评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。该脚本将在包含4个类别的合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线模型。输出混淆矩阵 (Confusion Matrix)。
2. **中等。** 将 `summarize` 替换为 `[mean, var, skew, kurtosis]`。在同一合成数据集上，四阶矩池化 (4-moment pooling) 的表现是否优于均值+方差？
3. **困难。** 使用 `torchaudio`，在 ESC-50 数据集的第1折上训练一个二维 CNN。报告5折交叉验证 (5-fold Cross-validation) 的准确率。加入 SpecAugment（时间掩码 time mask = 20，频率掩码 freq mask = 10）并报告性能变化 (delta)。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频界的 ImageNet | Google 发布的包含200万个片段、632个类别的弱标注 (Weakly-labeled) YouTube 数据集。 |
| ESC-50 | 小型分类基准测试 | 包含50个类别 × 40个片段的环境声音数据集。 |
| AST | 音频频谱图变换器 | 基于对数梅尔频谱块 (log-mel patches) 的视觉变换器 (ViT)；2021年 SOTA。 |
| BEATs | 自监督音频模型 | 微软推出的模型，截至2026年，其 iter3 版本在 AudioSet 上保持领先。 |
| Mixup | 样本对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩码的增强 | 将频谱图中随机的时间和频率区域置零。 |
| mAP | 多标签任务核心指标 | 跨类别和阈值的平均精度均值 (Mean Average Precision)。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021至2024年间的标杆架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024年及以后的默认选择。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主流的音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 历经时间考验的50类基准数据集。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 包含632个类别的 YouTube 分类体系 (Taxonomy)；至今仍是黄金标准。