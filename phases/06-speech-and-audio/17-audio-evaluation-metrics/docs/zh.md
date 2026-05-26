# 音频评估 — WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 无法衡量，便无法交付。本课程明确了 2026 年各项音频任务的评估指标：自动语音识别（ASR，含词错误率 WER、字符错误率 CER、实时因子 RTFx）、文本转语音（TTS，含平均意见得分 MOS、无参考 UTMOS、说话人嵌入余弦相似度 SECS、ASR 往返词错误率）、音频语言模型（MMAU、LongAudioBench）、音乐生成（FAD、CLAP）以及说话人相关任务（等错误率 EER）。此外还涵盖了用于模型横向对比的开放排行榜。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 6 阶段 · 04, 06, 07, 09, 10；第 2 阶段 · 09（模型评估）
**Time:** 约 60 分钟

## 问题背景

每项音频任务都包含多个评估指标，各自衡量不同的性能维度。选错指标，就会导致模型在开发仪表盘中表现优异，却在生产环境中惨不忍睹。以下是 2026 年的标准指标清单：

| 任务 | 核心指标 | 辅助指标 |
|------|---------|-----------|
| 自动语音识别（ASR） | WER | CER · RTFx · 首词延迟（first-token latency） |
| 文本转语音（TTS） | MOS / UTMOS | SECS · ASR 往返 WER · CER · TTFA |
| 语音克隆 | SECS（ECAPA 余弦相似度） | MOS · CER |
| 说话人验证 | EER | minDCF · 工作点处的 FAR / FRR |
| 说话人分离（Diarization） | DER | JER · 说话人混淆率 |
| 音频分类 | top-1 · mAP | 宏平均 F1 · 各类别召回率 |
| 音乐生成 | FAD | CLAP · 人工听评 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式语音到语音（Streaming S2S） | 延迟 P50/P95 | WER · MOS |

## 核心概念

![音频评估矩阵——指标 vs 任务 vs 2026 排行榜](../assets/eval-landscape.svg)

### ASR（自动语音识别，Automatic Speech Recognition）指标

**词错误率（Word Error Rate）。** `(S + D + I) / N`。评分前需将文本转为小写、去除标点符号并规范化数字。可使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。&lt; 5% 表示达到人类朗读语音的同等水平。

**字错误率（Character Error Rate）。** 公式相同，但基于字符级别计算。适用于中文（普通话、粤语）等存在分词歧义的声调语言。

**实时率倒数（inverse real-time factor）。** 每秒物理时钟时间（wall-clock）处理的音频秒数。数值越高越好。Parakeet-TDT 可达 3380×，Whisper-large-v3 约为 30×。

**首词元延迟（First-token latency）。** 从输入音频到输出首个转录词元（token）的物理时钟时间。对流式处理至关重要。Deepgram Nova-3 约为 150 ms。

### TTS（文本转语音，Text-to-Speech）指标

**平均意见得分（Mean Opinion Score）。** 1-5 分的人工评分。作为黄金标准但耗时较长。每个样本需收集 20 名以上听众的评分，每个模型需测试 100 个以上样本。

**UTMOS（2022-2026）。** 基于学习的平均意见得分预测模型。在标准基准测试中与人工评分的相关系数约为 0.9。F5-TTS 的 UTMOS 为 3.95，真实值（ground truth）为 4.08。

**说话人编码器余弦相似度（Speaker Encoder Cosine Similarity）。** 用于语音克隆。计算参考音频与克隆输出音频的 ECAPA 嵌入向量余弦相似度。&gt; 0.75 表示克隆语音可被识别。

**ASR 往返词错误率（WER-on-ASR-round-trip）。** 将 Whisper 应用于 TTS 输出音频，并针对原始输入文本计算词错误率。用于捕捉可懂度退化问题。2026 年当前最佳水平（State of the Art）：&lt; 2% 字错误率。

**首音频延迟（time-to-first-audio）。** 物理时钟时间延迟。Kokoro-82M 约为 100 ms；F5-TTS 约为 1 s。

### 语音克隆专属指标

将 **说话人编码器余弦相似度 + 平均意见得分 + 字错误率** 作为三元组进行评估。若余弦相似度得分高但平均意见得分低，说明音色正确但听感不自然；反之则说明听感自然但说话人特征不符。

### 说话人验证

**等错误率（Equal Error Rate）。** 误接受率（False Accept Rate）等于误拒绝率（False Reject Rate）时的阈值。ECAPA 在 VoxCeleb1-O 数据集上的等错误率为 0.87%。

**最小检测代价（min Detection Cost）。** 在选定工作点（通常误接受率=0.01）下的加权代价。相比等错误率更贴近实际生产环境。

### 说话人分离（Diarization）

**说话人日志错误率（Diarization Error Rate）。** `(FA + Miss + Confusion) / total_speaker_time`。包含漏检语音、误报语音和说话人混淆，各项均以占比形式计算。AMI 会议场景下，该错误率约为 10-20% 属于合理范围。pyannote 3.1 结合 Precision-2 商业版在录音质量良好的音频上可实现 &lt;10% 的错误率。

**杰卡德错误率（Jaccard Error Rate）。** 说话人日志错误率的替代指标，对短片段偏差具有更强的鲁棒性。

### 音频分类

多标签分类：所有类别的 **平均精度均值（mean Average Precision）**。AudioSet 数据集上，BEATs-iter3 的平均精度均值为 0.548。

多类互斥分类：**Top-1 与 Top-5 准确率**。Speech Commands v2 数据集上，Audio-MAE 的 Top-1 准确率为 99.0%。

类别不平衡场景：**宏平均 F1 分数（macro F1）** + **各类别召回率（per-class recall）**。需按类别单独报告——整体准确率会掩盖具体哪些类别表现不佳。

### 音乐生成

**弗雷歇音频距离（Fréchet Audio Distance）。** 真实音频与生成音频的 VGGish 嵌入分布之间的距离。MusicGen-small 在 MusicCaps 上的该距离为 4.5，MusicLM 为 4.0。数值越低越好。

**CLAP 得分（CLAP Score）。** 基于 CLAP 嵌入向量计算的文本-音频对齐得分。&gt; 0.3 表示对齐效果合理。

**人工听评平均意见得分（Listening panel MOS）。** 仍是消费级音乐质量的最终评判标准。Suno v5 在 TTS Arena 上的 ELO 评分为 1293（基于成对人工偏好得出）。

### 音频-语言基准测试

**大规模多音频理解（Massive Multi-Audio Understanding）。** 包含 1 万组音频问答对。

**MMAU-Pro。** 包含 1800 道高难度题目，分为四类：语音 / 环境音 / 音乐 / 多音频。四选一随机猜测正确率为 25%。Gemini 2.5 Pro 总体得分约 60%；所有模型在多音频任务上的平均得分约为 22%。

**LongAudioBench。** 包含数分钟长的音频片段及语义查询任务。Audio Flamingo Next 的表现优于 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 音频描述生成基准测试。采用 SPICE、CIDEr、FENSE 等评估指标。

### 流式语音到语音（Speech-to-Speech）

**延迟 P50 / P95 / P99。** 从用户语音结束到首次可听响应的物理时钟时间。Moshi 为 200 ms；GPT-4o Realtime 为 300 ms。

输出音频的 **词错误率 / 平均意见得分**。

**打断响应速度（Barge-in responsiveness）。** 从用户打断到助手静音的时间。目标值 &lt; 150 ms。

### 2026 年排行榜

| 排行榜 | 赛道/任务 | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | 英语 + 多语言 + 长音频 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，基于成对投票的 ELO 评分 | `artificialanalysis.ai/speech` |
| MMAU-Pro | 大型音频语言模型（LALM）推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU music subset | 音乐 LALM | (within MMAU) |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 构建

### 步骤 1：带文本归一化的词错误率 (Word Error Rate, WER)

from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17

### 步骤 2：语音合成 (Text-to-Speech, TTS) 往返词错误率 (WER)

def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)

### 步骤 3：用于语音克隆的说话人嵌入余弦相似度 (Speaker Embedding Cosine Similarity, SECS)

from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()

### 步骤 4：用于音乐生成的弗雷歇音频距离 (Fréchet Audio Distance, FAD)

from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")

### 步骤 5：用于说话人验证的等错误率 (Equal Error Rate, EER)（代码同第 6 课）

def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]

## 使用

每次部署都应搭配一个固定的评估框架 (evaluation harness)，在每次模型更新时运行。三条核心原则：

1. **评分前先进行归一化。** 转换为小写、去除标点符号、展开数字。明确报告所使用的归一化规则。
2. **报告分布情况，而非平均值。** 延迟指标报告 P50/P95/P99 分位数。分类任务报告各类别的召回率 (recall)。MMAU 指标按类别分别报告。
3. **运行一个标准的公开基准测试。** 即使你的生产数据有所不同，报告 Open ASR / TTS Arena / MMAU 的结果也能让评审人员进行公平对比。

## 常见陷阱

- **UTMOS 的外推问题。** 该模型在 VCTK 风格的干净语音上训练，对含噪/克隆/情感语音的评分效果较差。
- **平均意见得分 (Mean Opinion Score, MOS) 评审小组偏差。** 20 名 Amazon Mechanical Turk 众包工人 ≠ 20 名目标用户。如果项目风险较高，请付费聘请领域专家评审小组。
- **FAD 依赖于参考集。** 跨模型比较时，必须使用相同的参考分布。
- **聚合词错误率 (WER) 的误导性。** 整体 5% 的 WER 可能掩盖了带口音语音高达 30% 的 WER。应按人口统计学维度细分报告。
- **公开基准测试饱和。** 大多数前沿模型在标准基准测试上已接近性能天花板。构建一个能反映实际业务流量的内部保留集 (held-out set)。

## 发布

将文件保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型的发布挑选合适的指标、基准测试和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在示例输入上计算词错误率 (WER) / 字符错误率 (CER) / 等错误率 (EER) / 语音克隆相似度 (SECS) / 类弗雷歇音频距离 (FAD-ish) / 类多模态音频理解基准 (MMAU-ish)。
2. **中等。** 构建一个文本转语音 (TTS) 闭环词错误率评估框架。将你的 Kokoro 或 F5-TTS 输出送入 Whisper 模型。在 50 个提示词上计算 WER。标记 WER > 10% 的提示词。
3. **困难。** 在 MMAU-Pro 的语音与多音频子集（各 50 项）上，对你在第 10 课选择的大型音频语言模型 (LALM) 进行评分。报告各类别准确率，并与已公布的基准数值进行对比。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| WER | 自动语音识别 (ASR) 得分 | 归一化后词级别的 `(S+D+I)/N`。 |
| CER | 字符级 WER | 适用于声调语言或字符级系统。 |
| MOS | 人类主观评价 | 1-5 分制评分；需 20 名以上听众 × 100 个样本。 |
| UTMOS | 机器学习 MOS 预测器 | 训练所得模型；与人类 MOS 的相关性约为 0.9。 |
| SECS | 语音克隆相似度 | 参考音频与克隆音频之间的 ECAPA 余弦相似度。 |
| EER | 说话人验证得分 | 假接受率 (FAR) 等于假拒绝率 (FRR) 时的阈值。 |
| DER | 说话人日志 (Diarization) 得分 | （误报 + 漏报 + 混淆）/ 总时长。 |
| FAD | 音乐生成质量 | 基于 VGGish 嵌入的弗雷歇距离 (Fréchet Distance)。 |
| RTFx | 吞吐量 | 每秒实际运行时间生成的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 提供归一化实用工具的 WER/CER 计算库。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 基于学习的 MOS 预测模型。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成领域的标准评估指标。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 年实时排名榜单。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 基于人类投票的 TTS 排行榜。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理能力排行榜。
- [HEAR benchmark](https://hearbenchmark.com/) — 音频自监督学习 (SSL) 基准测试。