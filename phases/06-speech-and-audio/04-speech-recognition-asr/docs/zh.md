# 语音识别（Automatic Speech Recognition, ASR）—— CTC、RNN-T 与注意力机制（Attention）

> 语音识别本质上是对每个时间步的音频进行分类，并通过一个能够理解语言内容与静音片段的序列模型将它们连贯起来。CTC、RNN-T 和注意力机制是实现该任务的三种主流方法。选择其中一种，并透彻理解其设计动机。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 6 阶段 · 02（语谱图与梅尔频谱）、第 5 阶段 · 08（用于文本的卷积神经网络与循环神经网络）、第 5 阶段 · 10（注意力机制）
**预计耗时：** 约 45 分钟

## 核心挑战

你有一段 10 秒、采样率为 16 kHz 的音频片段，目标是将其转录为文本字符串："turn on the kitchen lights"。核心挑战在于结构层面：音频帧与字符之间并非一一对应。例如，单词 "okay" 的发音可能仅持续 200 毫秒，也可能长达 1200 毫秒。静音片段会穿插在语音流中，不同音素（phoneme）的时长也各不相同。此外，输出词元（token）的数量在推理前是未知的。

有三种主流建模方法可以解决这一问题：

1. **联结主义时序分类（Connectionist Temporal Classification, CTC）。** 输出包含特殊空白标记（blank）的逐帧词元概率。在解码阶段合并重复项与空白标记。非自回归（non-autoregressive），推理速度快。被 wav2vec 2.0、MMS 等模型采用。
2. **循环神经网络转换器（Recurrent Neural Network Transducer, RNN-T）。** 联合网络根据编码器输出的当前帧与历史词元预测下一个词元。支持流式（streamable）推理。被 Google 的端侧 ASR 系统、NVIDIA Parakeet 等采用。
3. **注意力编码器-解码器架构（Attention encoder-decoder）。** 编码器将音频压缩为隐藏状态（hidden states），解码器通过交叉注意力（cross-attention）自回归地生成词元。被 Whisper、SeamlessM4T 等模型采用。

截至 2026 年，LibriSpeech test-clean 数据集上的当前最优（State-of-the-Art, SOTA）词错误率（Word Error Rate, WER）分别为 1.4%（NVIDIA Parakeet-TDT-1.1B）和 1.58%（Whisper-Large-v3-turbo）。两者在精度上的差异微乎其微，但在部署层面的差异却极为显著。

## 核心概念

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**连接时序分类 (CTC) 的直观理解。** 假设编码器输出 `T` 个帧级分布，覆盖 `V+1` 个词元（V 个字符 + 空白符）。对于长度为 `U < T` 的目标字符串 `y`，任何能够折叠为 `y` 的帧对齐路径均有效。CTC 损失函数会对所有此类对齐路径求和。推理阶段：逐帧取 argmax，合并连续重复项，并移除空白符。

优势：非自回归 (non-autoregressive)、支持流式处理 (streamable)、零前瞻 (zero lookahead)。缺点：*条件独立性假设 (conditional independence assumption)* —— 每一帧的预测相互独立，因此模型内部不包含语言模型。可通过束搜索 (beam search) 或浅层融合 (shallow fusion) 引入外部语言模型 (LM) 来弥补。

**循环神经网络转录器 (RNN-T) 的直观理解。** 引入了一个*预测器 (predictor)* 网络来嵌入历史词元，以及一个*联合器 (joiner)* 将预测器状态与编码器帧结合，生成覆盖 `V+1` 的联合分布（其中的 `+1` 表示空操作/不输出）。它显式地建模了 CTC 所忽略的条件依赖性。由于每一步仅依赖于过去的帧和过去的词元，因此天然支持流式处理。

优势：支持流式处理 + 内置语言模型。缺点：训练更为复杂且显存占用高（涉及三维损失网格 (3D loss lattice)）；RNN-T 损失核函数 (loss kernels) 本身已发展成一个独立的库类别。

**注意力编码器-解码器 (Attention Encoder-Decoder)。** 编码器（6-32 层 Transformer）处理对数梅尔频谱 (log-mel) 帧。解码器（6-32 层 Transformer）通过交叉注意力机制 (cross-attention) 关注编码器输出，以自回归 (autoregressive) 方式生成词元。无对齐约束——注意力机制可以关注音频中的任意位置。除非对注意力范围进行限制（如分块处理的 Whisper-Streaming, 2024），否则不支持流式处理。

优势：在离线自动语音识别 (ASR) 任务中质量最高，且易于使用标准的序列到序列 (seq2seq) 工具链进行训练。缺点：自回归延迟与输出长度成正比；若不进行专门的工程优化，则无法实现流式处理。

### 词错误率 (WER)：核心指标

**词错误率 (Word Error Rate)** = `(S + D + I) / N`，其中 S 代表替换 (substitutions)，D 代表删除 (deletions)，I 代表插入 (insertions)，N 为参考文本的词数。它在词级别上等价于莱文斯坦编辑距离 (Levenshtein edit distance)。数值越低越好。WER 超过 20% 通常意味着模型不可用；低于 5% 则在朗读语音任务上达到人类水平。以下是 2026 年在标准基准测试上的数据：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 规模 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

上述模型均基于编码器-解码器架构或 RNN-T。纯 CTC 系统（如 wav2vec 2.0）在 test-clean 数据集上的表现约为 1.8%–2.1%。

## 动手构建

### 步骤 1：贪婪 CTC 解码 (Greedy CTC Decode)

def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out

遵循两条核心规则：合并连续重复的字符，并移除空白符（blank）。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：束搜索 CTC 解码 (Beam-Search CTC)

def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]

实际生产环境中通常采用结合语言模型（Language Model, LM）融合的前缀树束搜索（Prefix Tree Beam Search）；此处仅展示其概念骨架。

### 步骤 3：词错误率 (Word Error Rate, WER)

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))

### 步骤 4：基于 Whisper 的推理 (Inference)

import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])

仅需一行代码即可调用 2026 年最强的通用自动语音识别（Automatic Speech Recognition, ASR）模型。在 24 GB 显存的 GPU 上运行，推理速度约为实时音频的 20 倍。

### 步骤 5：使用 Parakeet 或 wav2vec 2.0 进行流式处理 (Streaming)

from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))

流式自动语音识别（Streaming ASR）需要分块编码器注意力机制（Chunked Encoder Attention）与状态延续（Carryover State）；请使用支持该特性的库（例如 Parakeet 的 NeMo，或配置了 `chunk_length_s` 参数的 `transformers` pipeline）。

## 实际应用指南

2026 年技术栈选型：

| 应用场景 | 推荐方案 |
|-----------|------|
| 英语、离线、追求最高质量 | Whisper-large-v3-turbo |
| 多语言、高鲁棒性 | SeamlessM4T v2 |
| 流式处理、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘设备、移动端、延迟 <500 ms | 量化版 Whisper-Tiny 或 Moonshine (2024) |
| 长音频处理 | 结合基于语音活动检测（Voice Activity Detection, VAD）分块的 Whisper (WhisperX) |
| 垂直领域（医疗、法律等） | 微调 wav2vec 2.0 + 领域语言模型融合 |

## 2026 年仍常见的实践陷阱

- **不使用语音活动检测（VAD）。** 在静音片段上运行 Whisper 会产生幻觉（例如输出“Thanks for watching!”）。务必使用 VAD 进行门控过滤。
- **字符级、词级与子词级词错误率（WER）。** 应在标准化（转为小写、去除标点符号）*之后*报告词级 WER。
- **语言识别（LID）漂移。** Whisper 的自动语言识别功能可能会将嘈杂的音频片段错误路由至日语或威尔士语；在已知语言的情况下，请强制指定 `language="en"`。
- **长音频未分块处理。** Whisper 的处理窗口为 30 秒。对于更长的音频，请使用 `chunk_length_s=30, stride=5` 参数。

## 部署上线

保存为 `outputs/skill-asr-picker.md`。针对特定的部署目标，选择合适的模型、解码策略、分块方式以及语言模型融合（LM fusion）方案。

## 练习

1. **简单。** 运行 `code/main.py`。该脚本会对人工构造的联结主义时序分类（CTC）输出进行贪心解码，并计算其相对于参考文本的词错误率（WER）。
2. **中等。** 正确实现第 2 步中的前缀树束搜索（prefix-tree beam search）（需考虑空白符合并规则）。在包含 10 个样本的合成数据集上，将其与贪心解码进行对比。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 数据集上使用 `whisper-large-v3-turbo` 模型。计算前 100 条语音的 WER，并与已发表的基准数据进行对比。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | 空白符损失 | 对所有帧到词元（token）对齐路径的边缘概率求和；非自回归（non-AR）。 |
| RNN-T | 流式损失 | CTC + 下一词元预测器；可处理词序问题。 |
| Attention enc-dec | Whisper 风格架构 | 编码器 + 交叉注意力解码器；离线场景下质量最佳。 |
| WER | 你报告的指标 | 词级计算公式为 `(S+D+I)/N`。 |
| Blank | 空状态 | CTC 中的特殊词元，用于表示“当前帧无输出”。 |
| LM fusion | 外部语言模型 | 在束搜索（beam search）过程中加入加权后的语言模型对数概率。 |
| VAD | 静音门控 | 语音活动检测器；用于裁剪非语音片段。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 的奠基论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 的原始论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年的经典论文；2024 年推出了 v3-turbo 扩展版本。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年 Open ASR Leaderboard 榜单第一名。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 涵盖 25+ 模型的实时基准测试平台。