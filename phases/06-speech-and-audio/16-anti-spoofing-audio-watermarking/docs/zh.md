# 语音防伪（Voice Anti-Spoofing）与音频水印（Audio Watermarking）—— ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆（Voice Cloning）的落地速度远超防御技术。2026 年的生产级语音系统必须具备两项核心能力：一是能够区分真实与伪造语音的检测器（Detector，如 AASIST、RawNet2），二是能够经受压缩和编辑处理的水印（Watermark，如 AudioSeal）。两者必须同时部署，否则不应上线语音克隆功能。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 6 阶段 · 06（说话人识别 Speaker Recognition）、第 6 阶段 · 08（语音克隆 Voice Cloning）
**Time:** 约 75 分钟

## 核心问题

三项相关防御技术：

1. **防伪/深度伪造检测（Anti-spoofing / Deepfake Detection）。** 给定一段音频，判断其为合成还是真实？ASVspoof 基准测试（ASVspoof 2019 → 2021 → 5）是该领域的黄金标准。
2. **音频水印（Audio Watermarking）。** 在生成的音频中嵌入人耳无法察觉的信号，以便检测器后续提取。AudioSeal（Meta）和 WavMark 是目前的开源选项。
3. **认证溯源（Authenticated Provenance）。** 对音频文件及元数据进行加密签名。遵循 C2PA / 内容真实性倡议（Content Authenticity Initiative）。

检测技术用于应对不配合的恶意攻击者。水印技术则用于满足合规要求——AI 生成的音频必须能够被明确标识。在 2026 年，这两者均为必备能力。

## 核心概念

![反欺骗（Anti-spoofing） vs 水印（Watermarking） vs 溯源（Provenance） — 三层防御机制](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 基准测试（Benchmark）

与以往版本相比，最大的变化在于：

- **众包数据（Crowdsourced data）**（非录音棚纯净音频）—— 贴近真实场景。
- **约 2000 名说话人**（此前约为 100 名）。
- **32 种攻击算法**。涵盖文本转语音（Text-to-Speech, TTS）+ 语音转换（Voice Conversion）+ 对抗性扰动（Adversarial Perturbation）。
- **两个赛道**。独立检测的反欺骗对策（Countermeasure, CM）；面向生物识别系统的抗欺骗自动说话人验证（Spoofing-robust Automatic Speaker Verification, SASV）。

ASVspoof 5 上的当前最佳（State-of-the-Art, SOTA）性能：等错误率（Equal Error Rate, EER）约为 7.23%。在较早的 ASVspoof 2019 LA 数据集上为 0.42% EER。实际部署时：在真实环境（in-the-wild）音频片段上，预期 EER 在 5% 到 10% 之间。

### AASIST 与 RawNet2 —— 检测模型家族

**AASIST**（2021 年发布，持续更新至 2026 年）。基于频谱特征（Spectral Features）的图注意力（Graph Attention）机制。目前在 ASVspoof 5 反欺骗对策任务中达到 SOTA 水平。

**RawNet2**。原始波形（Raw Waveform）卷积前端 + 时延神经网络（Time Delay Neural Network, TDNN）主干网络。作为更简单的基线模型，经过微调后仍具竞争力。

**NeXt-TDNN + 自监督学习（Self-Supervised Learning, SSL）特征**。2025 年变体：采用 ECAPA 架构风格 + WavLM 特征 + 焦点损失（Focal Loss）。在 ASVspoof 2019 LA 上实现了 0.42% 的 EER。

### AudioSeal —— 2024 年默认水印方案

Meta 推出的 **AudioSeal**（2024 年 1 月发布，2024 年 12 月更新至 v0.2）。核心设计如下：

- **局部化（Localized）**。以 16 kHz 采样分辨率（1/16000 秒）逐帧检测水印。
- **生成器与检测器联合训练（Joint Training）**。生成器学习嵌入人耳不可听的信号；检测器通过数据增强（Data Augmentation）学习识别该信号。
- **鲁棒性强**。可抵御 MP3/AAC 压缩、均衡器（Equalizer, EQ）调整、±10% 变速以及信噪比（Signal-to-Noise Ratio, SNR）为 +10 dB 的噪声混合。
- **速度快**。检测器运行速度达实时 485 倍；比 WavMark 快 1000 倍。
- **容量大**。每次语音片段（Utterance）可嵌入 16 位有效载荷（Payload），用于编码模型 ID、生成时间戳、用户 ID 等信息。

### WavMark

AudioSeal 出现前的开源基线方案。基于可逆神经网络（Invertible Neural Network），容量为 32 比特/秒。存在的问题：

- 同步暴力搜索（Brute-force Synchronization）速度缓慢。
- 易被高斯噪声或 MP3 压缩破坏/移除。
- 不满足实时处理需求。

### WaveVerify（2025 年 7 月）

针对 AudioSeal 的弱点进行了改进——特别是针对时间轴操作（Temporal Manipulations，如音频反转、变速）。采用基于特征线性调制（Feature-wise Linear Modulation, FiLM）的生成器 + 混合专家（Mixture-of-Experts, MoE）检测器。在标准攻击下性能与 AudioSeal 相当，且能有效处理时间轴编辑。

### 攻击者利用的防御缺口

根据 AudioMarkBench 的测试结果：“在音调偏移（Pitch Shift）攻击下，所有水印的比特恢复准确率（Bit Recovery Accuracy）均低于 0.6，表明水印几乎被完全移除。”**音调偏移是通用攻击手段。** 截至 2026 年，尚无任何水印方案能完全抵御激进的音调修改。这也是为何必须将检测模型（如 AASIST）与水印技术结合使用的原因。

### C2PA / 内容真实性倡议（Content Authenticity Initiative）

这并非机器学习技术，而是一种清单格式（Manifest Format）。音频文件携带经加密签名的元数据（Cryptographically Signed Metadata），记录创作工具、作者及日期等信息。Audobox 与 Seamless 等项目已采用该标准。适用于溯源；但如果恶意行为者重新编码（Re-encode）并剥离元数据，该机制将失效。

## 动手构建

### 步骤 1：一个简单的频谱特征检测器（示例）

def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92

合成语音（Synthetic Speech）的高频能量通常异常平坦。生产环境中的检测器会采用 AASIST，而非此方法。但其核心思路是成立的。

### 步骤 2：AudioSeal 嵌入与检测

from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload

### 步骤 3：评估 —— 等错误率（Equal Error Rate, EER）

def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]

### 步骤 4：生产环境集成

def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest

每次生成输出均会包含：(1) 水印（Watermark），(2) 已签名的清单文件（Signed Manifest），(3) 符合数据保留策略的审计日志（Audit Log）。

## 实际应用

| 应用场景 | 防御策略 |
|----------|---------|
| 部署文本转语音（Text-to-Speech, TTS）/ 语音克隆（Voice Cloning）模型 | 对所有输出强制嵌入 AudioSeal 水印（不可妥协） |
| 生物特征声纹解锁（Biometric Voice Unlock） | AASIST + ECAPA 模型集成；活体检测挑战（Liveness Challenge） |
| 呼叫中心欺诈检测（Fraud Detection） | 对 20% 的呼入语音样本进行 AASIST 检测 |
| 播客（Podcast）真实性验证 | 上传时进行 C2PA 签名，若为 AI 生成则附加 AudioSeal |
| 研究 / 训练检测器 | 使用 ASVspoof 5 的训练集/开发集/评估集 |

## 潜在陷阱

- **仅添加水印却从不运行检测器。** 毫无意义。请将检测器集成到你的持续集成（CI）流程中。
- **未经校准的检测。** 在 ASVspoof LA 数据集上训练的 AASIST 模型容易过拟合；实际场景中的准确率会下降。请在你的业务领域数据上进行校准。
- **音高偏移漏洞。** 强烈的音高偏移会破坏大多数水印。请准备备用的检测方案。
- **元数据剥离与重新托管。** 仅靠重新编码即可轻易绕过 C2PA 标准。务必同时结合密码学防御与感知防御（水印）。
- **将活体检测作为防伪手段。** 要求用户朗读随机短语。这能防止重放攻击，但无法抵御实时克隆。

## 部署上线

保存为 `outputs/skill-spoof-defender.md`。为语音生成（voice-gen）部署选择合适的检测模型、水印方案、来源清单（provenance manifest）以及运维手册。

## 练习

1. **简单。** 运行 `code/main.py`。在合成音频上测试简易检测器与简易水印的嵌入/提取功能。
2. **中等。** 安装 `audioseal`，在文本转语音（TTS）输出中嵌入 16 位有效载荷（payload），然后重新解码。向音频添加噪声干扰，并测量比特恢复准确率（Bit Recovery Accuracy）。
3. **困难。** 在 ASVspoof 2019 LA 数据集上微调 RawNet2 或 AASIST 模型。测量等错误率（EER）。在预留的 F5-TTS 生成音频片段集上进行测试——观察分布外（OOD）检测性能如何下降。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| ASVspoof | 基准测试 | 两年一度的挑战赛；2024 年为 ASVspoof 5。 |
| CM（countermeasure） | 检测器 | 分类器：用于区分真实语音与合成/转换语音。 |
| SASV | 说话人验证 + 反制措施 | 集成生物特征识别与反欺骗检测。 |
| AudioSeal | Meta 水印 | 本地化部署，16 位有效载荷，速度比 WavMark 快 485 倍。 |
| Bit Recovery Accuracy | 水印存活率 | 遭受攻击后成功恢复的有效载荷比特比例。 |
| C2PA | 来源清单 | 关于内容创建/作者身份的加密元数据。 |
| AASIST | 检测器系列 | 基于图注意力机制的反欺骗检测最先进（SOTA）模型。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前基准测试。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 默认水印方案。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 针对时序攻击的混合专家（MoE）检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — 最先进的（SOTA）检测骨干网络。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估基准。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — 来源清单格式规范。