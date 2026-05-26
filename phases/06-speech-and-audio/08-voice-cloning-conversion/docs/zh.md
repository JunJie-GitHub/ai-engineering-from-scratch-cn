# 语音克隆 (Voice Cloning) 与语音转换 (Voice Conversion)

> 语音克隆使用他人的声音朗读你的文本。语音转换在保留你所说内容的前提下，将你的声音转换为他人的声音。两者都依赖于同一个底层原语 (Primitive)：将说话人身份与语音内容分离。

**类型：** 构建
**语言：** Python
**前置条件：** 第 6 阶段 · 06（说话人识别 (Speaker Recognition)），第 6 阶段 · 07（文本转语音 (TTS)）
**耗时：** 约 75 分钟

## 问题背景

到 2026 年，仅需一段 5 秒的音频片段，配合消费级 GPU 即可生成任何人声音的高质量克隆。ElevenLabs、F5-TTS、OpenVoice v2 和 VoiceBox 等模型均已支持零样本 (Zero-shot) 或少样本 (Few-shot) 克隆功能。这项技术既是福音（无障碍文本转语音、配音、辅助发声），也是武器（诈骗电话、政治深度伪造 (Deepfakes)、知识产权盗窃）。

两项密切相关的任务：

- **语音克隆（TTS 端）：** 文本 + 5 秒参考语音 → 生成该声音的音频。
- **语音转换（语音端）：** 源音频（人物 A 说 X）+ 人物 B 的参考语音 → 生成人物 B 说 X 的音频。

两者均将波形分解为（内容、说话人、韵律）三个要素，并将一个源的内容与另一个源的说话人特征重新组合。

2026 年产品交付时必须遵守的关键约束：**欧盟（《人工智能法案》，2026 年 8 月强制执行）和加利福尼亚州（AB 2905 法案，2025 年生效）已依法强制要求添加水印 (Watermarking) 与设置同意验证机制 (Consent Gates)**。你的处理流水线 (Pipeline) 必须输出不可听水印，并拒绝未经授权的克隆请求。

## 核心概念

![语音克隆与语音转换：因子分解、替换说话人、重新组合](../assets/voice-cloning.svg)

**零样本克隆（Zero-shot cloning）。** 将一段 5 秒的音频片段输入至已在数千名说话人数据上训练过的模型。说话人编码器（speaker encoder）会将该片段映射为说话人嵌入向量（speaker embedding）；文本到语音（TTS）解码器则基于该嵌入向量与输入文本进行条件生成。

应用模型：F5-TTS (2024)、YourTTS (2022)、XTTS v2 (2024)、OpenVoice v2 (2024)。

**少样本微调（Few-shot fine-tuning）。** 录制目标声音 5 到 30 分钟。使用 LoRA 对基础模型进行约一小时的微调。音质将从“尚可”跃升至“难以分辨”。Coqui 和 ElevenLabs 均支持此模式；社区也常将其应用于 F5-TTS。

**语音转换（Voice Conversion, VC）。** 主要分为两大流派：

- **识别-合成（Recognition-synthesis）。** 运行类似自动语音识别（ASR）的模型以提取内容表征（例如软音素后验概率、音素后验图 PPGs），随后结合目标说话人嵌入向量重新合成语音。该方法对语言和口音具有较强的鲁棒性。代表模型：KNN-VC (2023)、Diff-HierVC (2023)。
- **解耦（Disentanglement）。** 训练一个自编码器（autoencoder），在瓶颈层（bottleneck）的潜在空间（latent space）中将内容、说话人特征和韵律分离。在推理阶段替换说话人嵌入向量即可。该方法质量略低但速度更快。代表模型：AutoVC (2019) 及 VITS-VC 变体。

**基于神经编解码器的克隆（Neural codec-based cloning，2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox 等模型将音频视为来自 SoundStream / EnCodec 的离散词元（discrete tokens），并基于这些编解码器词元训练大型自回归（autoregressive）或流匹配（flow-matching）模型。在短提示词下，其音质可与 ElevenLabs 相媲美。

### 伦理考量：绝非事后附加

**水印技术（Watermarking）。** PerTh (Perth) 与 SilentCipher (2024) 可在音频中不可察觉地嵌入约 16-32 位的标识符。该水印能够经受重新编码、流媒体传输及常规剪辑的考验。目前已提供可用于生产环境的开源实现。

**授权验证（Consent gates）。** 必须将每次克隆输出与可验证的授权记录绑定。例如：“本人 Rohit，于 2026-04-22 授权将此声音用于 X 用途。”记录需存储于防篡改日志（tamper-evident log）中。

**检测技术（Detection）。** AASIST、RawNet2 和 Wav2Vec2-AASIST 均作为检测器发布。ASVspoof 2025 挑战赛公布的数据显示，针对 ElevenLabs、VALL-E 2 和 Bark 的输出，最先进检测器的等错误率（EER）已降至 0.8%–2.3%。

### 性能指标（2026）

| 模型 | 是否零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

对于大多数听众而言，SECS > 0.70 的语音通常已与目标声音难以分辨。

## 动手构建

### 步骤 1：使用识别-合成（recognition-synthesis）进行分解（`main.py` 中的纯代码演示）

def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)

概念上很简单；代码主体集中在 `tts_model` 和说话人编码器（speaker encoder）中。

### 步骤 2：使用 F5-TTS 进行零样本克隆（zero-shot clone）

from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)

参考文本（reference transcript）必须与音频完全匹配；不匹配会导致对齐（alignment）失败。

### 步骤 3：使用 KNN-VC 进行声音转换（voice conversion）

import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])

KNN-VC 调用 WavLM 提取源音频与目标音频池（target pool）中每一帧的嵌入向量（embeddings），随后将源音频的每一帧替换为池中的最近邻。该方法属于非参数化（non-parametric）模型，仅需一分钟目标语音即可运行。

### 步骤 4：嵌入水印（watermark）

from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes

有效载荷（payload）约为 32 位，在经过 MP3 重新编码和轻微噪声干扰后仍可被检测到。

### 步骤 5：同意验证门控（consent gate）

def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav

## 实际应用

2026 年技术栈（stack）：

| 场景 | 推荐方案 |
|-----------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业级生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 声音转换（重写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调（fine-tune） | StyleTTS 2 + 说话人适配器（speaker adapter） |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测（deepfake detection） | Wav2Vec2-AASIST |

## 常见陷阱

- **参考文本未对齐。** F5-TTS 及类似模型要求参考文本必须与参考音频完全一致，包括标点符号。
- **参考音频混响过重。** 回声会严重破坏克隆效果。请在无混响环境下近距离录音。
- **情感不匹配。** 使用“欢快”的参考音频进行训练，会导致所有克隆输出都带有欢快的情感。请确保参考音频的情感与目标使用场景相匹配。
- **语言特征泄露（language leakage）。** 克隆英语说话人后让模型说法语，通常仍会带有英语口音；请使用跨语言模型（如 XTTS、VALL-E X）。
- **未添加水印。** 自 2026 年 8 月起，在欧盟地区未加水印的语音产品将无法合法发布。

## 部署发布

保存为 `outputs/skill-voice-cloner.md`。设计一个包含同意验证门控（consent gate）、水印（watermark）和质量目标（quality target）的克隆或转换流水线（pipeline）。

## 练习

1. **简单。** 运行 `code/main.py`。通过计算交换前后两个“说话人”之间的余弦相似度（cosine similarity），演示说话人嵌入（speaker-embedding）的替换。
2. **中等。** 使用 OpenVoice v2 克隆你自己的声音。测量参考音频与克隆音频之间的 SECS（说话人嵌入余弦相似度）。通过 Whisper 测量 CER（字符错误率）。
3. **困难。** 将 SilentCipher 水印应用于 20 个克隆样本，对其进行 128 kbps MP3 编码与解码处理，并检测有效载荷（payload）。报告比特准确率（bit-accuracy）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 零样本克隆（Zero-shot clone） | 5秒音频就够用 | 预训练模型 + 说话人嵌入（speaker embedding）；无需额外训练。 |
| 音素后验图（PPG） | 音素后验图 | 每帧自动语音识别（ASR）后验概率，用作与语言无关的内容表示（content representation）。 |
| K近邻语音转换（KNN-VC） | 最近邻转换 | 用目标池中最接近的帧替换每个源帧。 |
| 神经编解码器语音合成（Neural codec TTS） | VALL-E 风格 | 基于 EnCodec/SoundStream 词元（tokens）的自回归（AR）模型。 |
| 水印（Watermark） | 不可听签名 | 嵌入音频中的比特信息，可经受重新编码。 |
| 说话人嵌入余弦相似度（SECS） | 克隆保真度 | 目标说话人与克隆说话人嵌入向量之间的余弦值。 |
| 音频反欺骗系统（AASIST） | 深度伪造检测器 | 反欺骗（anti-spoof）模型；用于检测合成语音。 |

## 扩展阅读

- [Chen 等人 (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源的最先进（SOTA）零样本克隆模型。
- [Baevski 等人 / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 与 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 神经编解码器语音合成（neural-codec TTS）。
- [Qian 等人 (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦（disentanglement）的语音转换（voice conversion）。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的语音转换（VC）。
- [SilentCipher (2024) — 音频水印](https://github.com/sony/silentcipher) — 可投入生产的 32 位音频水印方案。
- [ASVspoof 2025 结果](https://www.asvspoof.org/) — 检测器与合成器之间的军备竞赛，更新至 2026 年。