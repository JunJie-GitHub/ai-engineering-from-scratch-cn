# 说话人识别与验证

> 自动语音识别（Automatic Speech Recognition, ASR）问的是“他们说了什么？”，而说话人识别（Speaker Recognition）问的是“这是谁说的？”。其数学原理看似相同——嵌入向量（Embedding）加余弦相似度（Cosine Similarity）——但每一项生产决策都取决于一个单一的等错误率（Equal Error Rate, EER）数值。

**类型:** 构建
**语言:** Python
**前置条件:** 第6阶段 · 02（语谱图与梅尔频谱）, 第5阶段 · 22（嵌入模型）
**耗时:** 约45分钟

## 问题描述

用户说出一段口令。你需要判断：这是否是他们声称的那个人（*验证（Verification）*，1:1），还是你注册库中的某个人（*识别（Identification）*，1:N）？或者两者都不是——这是一个未知的说话人（*开放集（Open-set）*）？

2018年之前：高斯混合模型-通用背景模型（Gaussian Mixture Model - Universal Background Model, GMM-UBM）结合 i-向量（i-vector）。等错误率（EER）表现尚可，但对信道偏移（Channel Shift，如手机与笔记本电脑的差异）和情绪变化较为敏感。2018–2022年：x-向量（x-vector，采用角度间隔（Angular Margin）训练的时延神经网络（Time Delay Neural Network, TDNN）主干）。2022年至今：ECAPA-TDNN 与 WavLM-large 嵌入向量。到2026年，该领域已由三种模型和一项指标主导。

该指标即为 **等错误率（Equal Error Rate, EER）**。设定决策阈值，使错误接受率（False Accept Rate, FAR）等于错误拒绝率（False Reject Rate, FRR）。两条曲线的交叉点即为 EER。该指标被广泛应用于每一篇学术论文、每一个评测排行榜以及每一次采购招标中。

## 核心概念

![包含嵌入、余弦相似度与等错误率的注册与验证流水线](../assets/speaker-verification.svg)

**流水线（Pipeline）。** 注册（Enrollment）：录制目标说话人 5–30 秒的语音；计算固定维度的嵌入向量（Embedding）（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证（Verification）：获取测试语音片段的嵌入向量；计算余弦相似度（Cosine Similarity）；与预设阈值进行比较。

**ECAPA-TDNN（2020 年提出，至 2026 年仍占主导地位）。** 强调通道注意力、传播与聚合的时延神经网络（Time-Delay Neural Network）。采用带挤压与激励（Squeeze-Excitation）模块的一维卷积块，结合多头注意力池化（Multi-Head Attention Pooling），最后通过线性层输出 192 维向量。在 VoxCeleb 1+2 数据集（2,700 名说话人，110 万条语音）上使用加性角度间隔损失（Additive Angular Margin Loss, AAM-softmax）进行训练。

**WavLM-SV（2022 年及以后）。** 使用 AAM 损失对预训练的 WavLM-large 自监督学习（Self-Supervised Learning, SSL）骨干网络进行微调。质量更高但速度较慢——模型大小超过 300 MB，而前者仅为 15 MB。

**x-vector（基线模型）。** 时延神经网络（TDNN）结合统计池化（Statistics Pooling）。经典架构；在 CPU 或边缘设备上依然具有实用价值。

**AAM-softmax。** 在标准 softmax 的基础上，于角度空间中引入间隔参数 `m`：针对正确类别计算 `cos(θ + m)`。该机制强制拉大类间角度间隔。典型参数为 `m=0.2`，缩放因子 `s=30`。

### 评分（Scoring）

- **余弦相似度（Cosine）：** 计算注册与测试嵌入向量之间的余弦值，基于阈值进行判定。
- **概率线性判别分析（Probabilistic Linear Discriminant Analysis, PLDA）：** 将嵌入向量投影至潜在空间，在该空间中同说话人与异说话人具有闭式似然比（Closed-Form Likelihood Ratio）。在余弦相似度基础上叠加使用，可进一步降低 10–20% 的等错误率（Equal Error Rate, EER）。2020 年之前的标准方案；目前仅用于闭集（Closed-Set）场景。
- **分数归一化（Score Normalization）：** 采用 `S-norm` 或 `AS-norm` 方法，将每个得分与一组冒充者（Imposter）的均值和标准差进行归一化。对于跨域评估（Cross-Domain Evaluation）至关重要。

### 关键性能指标（2026 年参考）

| 模型 | VoxCeleb1-O 等错误率 (EER) | 参数量 | 吞吐量 (A100) |
|-------|-----------------|--------|-------------------|
| x-vector（经典） | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 分割 + 嵌入 | 0.65% | 6 M | 100× RT |
| ReDimNet（2024） | 0.39% | 24 M | 100× RT |

### 说话人日志分析（Speaker Diarization）

用于解决多说话人音频片段中“谁在何时说话”的问题。标准流水线：语音活动检测（Voice Activity Detection, VAD）→ 分段 → 提取各段嵌入向量 → 聚类（层次聚类或谱聚类）→ 边界平滑。现代技术栈：`pyannote.audio` 3.1，通过单次调用即可集成说话人分割、嵌入提取与聚类功能。2026 年在 AMI 数据集上的最新最优（State-Of-The-Art, SOTA）说话人日志错误率（Diarization Error Rate, DER）约为 15%（较 2022 年的 23% 显著下降）。

## 动手实践

### 步骤 1：基于 MFCC 统计特征的简易嵌入向量 (toy embedding)

def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d

这远非当前最先进 (State-of-the-Art, SOTA) 的方案，仅用于教学演示。`code/main.py` 将其作为概念验证 (proof-of-concept)，应用于合成说话人数据。

### 步骤 2：余弦相似度 (cosine similarity) 与阈值 (threshold)

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold

### 步骤 3：基于相似度对计算等错误率 (Equal Error Rate, EER)

def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]

返回 `(eer, threshold_at_eer)`。请同时报告这两个指标。

### 步骤 4：使用 SpeechBrain 进行生产环境部署

from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data

### 步骤 5：使用 pyannote 进行说话人日志分析 (speaker diarization)

from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")

## 使用场景

2026 年技术栈 (stack) 选型：

| 场景 | 推荐方案 |
|-----------|------|
| 闭集 1:1 验证 (closed-set 1:1 verification)，边缘设备 (edge) | ECAPA-TDNN + 余弦阈值 |
| 开集验证 (open-set verification)，云端 | WavLM-SV + 自适应分数归一化 (AS-norm) |
| 说话人日志分析 (diarization)（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗 (anti-spoofing)（重放/深度伪造检测） | AASIST 或 RawNet2 |
| 微型嵌入式设备（关键词唤醒 (KWS) + 声纹注册 (enrollment)） | Titanet-Small (NeMo) |

## 常见陷阱 (Pitfalls)

- **信道不匹配（Channel mismatch）。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话通话音频。务必在目标信道（channel）上进行评估。
- **短语音片段（Short utterances）。** 测试音频低于 3 秒时，等错误率（EER）会急剧下降。
- **带噪注册（Enrollment with noise）。** 单个带噪注册样本会污染锚点（anchor）。请使用 ≥3 个干净样本并取平均。
- **跨条件固定阈值（Fixed threshold across conditions）。** 务必在来自目标域的独立开发集（dev set）上调整阈值。
- **对未归一化嵌入向量使用余弦相似度（Cosine on non-normalized embeddings）。** 请先进行 L2 归一化（L2-normalize）；否则向量模长将主导计算结果。

## 部署上线

保存为 `outputs/skill-speaker-verifier.md`。选定模型、注册协议（enrollment protocol）、阈值调优计划以及反欺诈防护措施。

## 练习

1. **简单。** 运行 `code/main.py`。构建合成的“说话人”（不同音色特征），执行注册，并在包含 100 对样本的试验列表（trial list）上计算等错误率（EER）。
2. **中等。** 在 30 条 VoxCeleb1 语音片段（5 位说话人 × 每人 6 条）上使用 SpeechBrain ECAPA。分别使用余弦相似度（cosine）与概率线性判别分析（PLDA）计算 EER。
3. **困难。** 使用 `pyannote.audio` 构建完整的“注册 → 说话人日志分割（diarize） → 验证”流水线。在 AMI 开发集上评估说话人日志错误率（DER）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| EER（等错误率） | 核心指标 | 错误接受率（False Accept）等于错误拒绝率（False Reject）时的阈值。 |
| Verification（说话人验证） | 1:1 比对 | “这是 Alice 吗？” |
| Identification（说话人识别） | 1:N 匹配 | “说话的人是谁？” |
| Open-set（开放集） | 可能存在未知对象 | 测试集中可能包含未注册的说话人。 |
| Enrollment（注册） | 录入信息 | 计算说话人的参考嵌入向量（reference embedding）。 |
| AAM-softmax（加性角度边际 Softmax） | 损失函数 | 带有加性角度边际的 Softmax；强制类簇分离。 |
| PLDA（概率线性判别分析） | 经典打分方法 | 概率 LDA；在嵌入向量之上进行似然比打分。 |
| DER（说话人日志错误率） | 日志评估指标 | 说话人日志错误率（Diarization Error Rate）—— 漏报（miss）+ 虚警（false alarm）+ 混淆（confusion）。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) —— 经典的深度嵌入向量（deep-embedding）论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) —— 2020 至 2026 年间的主流架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) —— 用于说话人验证（SV）与说话人日志分割的自监督学习（SSL）骨干网络。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) —— 工业级说话人日志分割与嵌入向量技术栈。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) —— 各模型当前的 EER 排名。