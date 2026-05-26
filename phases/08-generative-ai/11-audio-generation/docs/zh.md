# 音频生成 (Audio Generation)

> 音频是采样率为 16-48 kHz 的一维信号 (1-D Signal)。一段五秒的音频片段包含 8 万到 24 万个采样点。没有任何 Transformer 会直接对该序列进行注意力计算。到 2026 年，所有生产级音频模型的解决方案都是一致的：使用神经音频编解码器 (Neural Audio Codec，如 Encodec、SoundStream、DAC) 将音频压缩为 50-75 Hz 的离散词元 (Discrete Tokens)，再由 Transformer 或扩散模型 (Diffusion Model) 生成这些词元。

**类型：** 构建
**语言：** Python
**前置要求：** 第 6 阶段 · 02（音频特征），第 6 阶段 · 04（自动语音识别 ASR），第 8 阶段 · 06（去噪扩散概率模型 DDPM）
**预计耗时：** 约 45 分钟

## 核心问题

三大音频生成任务：

1. **文本转语音 (Text-to-Speech, TTS)。** 输入文本，生成语音。纯净语音属于窄带信号且具有强烈的音素结构——基于词元的 Transformer (Transformer-over-Tokens) 能很好地解决该问题。代表模型：VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成 (Music Generation)。** 输入提示词（文本、旋律、和弦进行或流派），生成音乐。其数据分布要广泛得多。代表模型：MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效/声音设计 (Audio Effects / Sound Design)。** 输入提示词，生成环境音或拟音 (Foley)。代表模型：AudioGen、AudioLDM 2、Stable Audio Open。

这三类任务均基于相同的底层架构：神经音频编解码器 + 词元自回归 (Token-Autoregressive, Token-AR) 或扩散生成器。

## 核心概念

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### 神经音频编解码器 (Neural Audio Codecs)

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器 (Convolutional Encoder) 将波形压缩为每个时间步的向量；残差矢量量化 (Residual Vector Quantization, RVQ) 将每个向量转换为 K 个码本索引 (Codebook Indices) 的级联序列。解码器则执行逆向操作。以 24 kHz 音频、2 kbps 码率为例，使用 8 个 RVQ 码本且帧率为 75 Hz 时，相当于每秒生成 600 个词元。

waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8

### 上层两大生成范式

**词元自回归 (Token-Autoregressive)。** 将 RVQ 词元展平为序列，输入仅解码器 Transformer (Decoder-Only Transformer)。MusicGen 采用“延迟并行 (Delayed Parallel)”策略，以各流特定的偏移量并行输出 K 个码本流。VALL-E 则根据文本提示词加 3 秒语音样本生成语音词元。

**潜在扩散 (Latent Diffusion)。** 将编解码器词元打包为连续潜在变量 (Continuous Latents)，或使用类别扩散 (Categorical Diffusion) 对其进行建模。Stable Audio 2.5 在连续音频潜在变量上应用流匹配 (Flow Matching)。AudioLDM 2 采用“文本到梅尔频谱再到音频 (Text-to-Mel-to-Audio)”的扩散流程。

2024-2026 年的技术趋势：流匹配在音乐生成领域占据优势（推理速度更快、生成样本更清晰），而词元自回归仍在语音生成领域占据主导地位，因为其天然具备因果性 (Causal) 且非常适合流式输出。

## 生产级应用全景

| 系统 | 任务 | 骨干网络 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | 文本转语音 (Text-to-Speech, TTS) | 令牌自回归 (Token-Autoregressive, Token-AR) + 神经声码器 (Neural Vocoder) | 首令牌约 300ms |
| OpenAI GPT-4o audio | 全双工语音 (Full-duplex speech) | 端到端多模态自回归 (End-to-End Multimodal Autoregressive) | ~200ms |
| NaturalSpeech 3 | 文本转语音 (TTS) | 潜在流匹配 (Latent Flow Matching) | 非流式 (Non-streaming) |
| Stable Audio 2.5 | 音乐 / 音效 (Sound Effects, SFX) | 扩散 Transformer (Diffusion Transformer, DiT) + 音频潜在空间流匹配 | 1 分钟片段约 10s |
| Suno v4 | 完整歌曲 | 未公开；疑似令牌自回归 | 每首歌曲约 30s |
| Udio v1.5 | 完整歌曲 | 未公开 | 每首歌曲约 30s |
| MusicGen 3.3B | 音乐 | 基于 32kHz EnCodec 的令牌自回归 | 实时 (Real-time) |
| AudioCraft 2 | 音乐 + 音效 (SFX) | 流匹配 (Flow Matching) | 5s 片段约 5s |
| Riffusion v2 | 音乐 | 频谱图扩散 (Spectrogram Diffusion) | ~10s |

## 动手实现

`code/main.py` 模拟了核心思想：在由两种不同“风格”生成的合成“音频令牌 (Audio Token)”序列上，训练一个小型的下一令牌预测 Transformer（风格 A 为高低令牌交替，风格 B 为单调递增）。根据风格条件进行采样。

### 步骤 1：合成音频令牌

def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]

### 步骤 2：训练小型令牌预测器

一个基于风格条件的二元语法 (Bigram) 风格预测器。其核心在于展示该工作流：编解码器 (Codec) 令牌 → 交叉熵训练 (Cross-Entropy Training) → 自回归采样 (Autoregressive Sampling)。

### 步骤 3：条件采样

给定风格令牌和起始令牌，从预测分布中采样下一个令牌。重复此过程 20-40 次以生成令牌序列。

## 常见陷阱

- **编解码器质量决定输出上限。** 如果编解码器无法高保真地还原声音，生成器质量再高也无济于事。DAC 是目前开源领域的最佳选择。
- **残差矢量量化 (Residual Vector Quantization, RVQ) 误差累积。** RVQ 的每一层都在对前一层的残差进行建模。第一层的误差会逐层传播。在较高层使用温度 (Temperature) 为 0 进行采样有助于缓解此问题。
- **音乐结构建模。** 在 75 Hz 的采样率下，30 秒的音频对应 2 万多个令牌，这对 Transformer 来说极具挑战。MusicGen 采用滑动窗口 (Sliding Window) + 提示词延续 (Prompt Continuation)；Stable Audio 则使用较短片段 + 交叉淡入淡出 (Crossfading)。
- **边界伪影 (Artifacts)。** 生成片段之间的交叉淡入淡出需要精细的重叠相加 (Overlap-Add) 处理。
- **对高质量数据的渴求。** 音乐生成模型需要数万小时的授权音乐数据。Suno / Udio 与美国唱片业协会 (RIAA) 的诉讼案（2024 年）将这一问题推向了风口浪尖。
- **声音克隆的伦理问题。** 仅需 3 秒的音频样本加上一段文本提示，VALL-E / XTTS / ElevenLabs 等模型即可完成声音克隆。所有投入生产的模型都必须配备滥用检测机制与退出名单 (Opt-out Lists)。

## 使用它

| 任务 | 2026 技术栈 |
|------|------------|
| 商用文本转语音 (TTS) | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 语音克隆（需授权验证） | XTTS v2（开源）或 ElevenLabs Pro |
| 快速生成背景音乐 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐生成 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 (Foley) | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音智能体 (Voice Agent) | GPT-4o realtime 或 Gemini Live |
| 开放权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0 或 AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## Ship It

保存 `outputs/skill-audio-brief.md`。该技能接收一份音频需求简报（包含任务、时长、风格、音色、授权许可），并输出以下内容：模型与托管方案、提示词格式（流派标签、风格描述符、结构标记）、编解码器 (Codec) + 生成器 (Generator) + 声码器 (Vocoder) 处理链、随机种子协议，以及评估方案（平均意见得分 (MOS) / CLAP 得分 / 文本转语音的词错误率 (CER) / 用户 A/B 测试）。

## Exercises

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成的序列是否与该风格的模式相匹配。
2. **中等。** 添加延迟并行解码 (Delayed Parallel Decoding)：模拟 2 个必须保持 1 步偏移的 Token 流。训练一个联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。使用三个不同的提示词生成一段 10 秒的音频片段；进行 A/B 测试以评估风格遵循度。

## Key Terms

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 编解码器 (Codec) | “神经压缩” | 音频的编码器/解码器；典型输出为 50-75 Hz 的 Token。 |
| 残差矢量量化 (RVQ) | “残差 VQ” | K 个量化器的级联；每个量化器对前一个的残差进行建模。 |
| 词元/标记 (Token) | “一个编解码符号” | 码本中的离散索引；通常为 1024 或 2048。 |
| 延迟并行 (Delayed Parallel) | “偏移码本” | 以交错偏移的方式输出 K 个 Token 流，以缩短序列长度。 |
| 流匹配 (Flow Matching) | “2024 年音频领域的突破” | 扩散模型的更直路径替代方案；采样速度更快。 |
| 语音提示词 (Voice Prompt) | “3 秒样本” | 说话人嵌入或 Token 前缀，用于引导克隆语音的生成。 |
| 梅尔频谱图 (Mel Spectrogram) | “可视化特征” | 对数幅度感知频谱图；被许多 TTS 系统采用。 |
| 声码器 (Vocoder) | “梅尔转波形” | 将梅尔频谱图转换回音频的神经网络组件。 |

## 生产环境提示：音频处理本质上是流式问题

在所有输出模态中，音频是用户唯一期望能够*边生成边输出*，而非一次性全部呈现的。在生产环境中，这意味着单输出 Token 耗时（Time Per Output Token）至关重要，因为系统的目标吞吐量需匹配用户的听觉处理速度，而非阅读速度。对于采样率为 16kHz、使用 Encodec 分词速率约为 75 tokens/second 的音频，服务器必须为每位用户维持 ≥75 tokens/sec 的生成速率，才能确保播放流畅。

这带来了两项架构层面的影响：

- **流匹配（Flow-matching）音频模型难以直接实现流式输出。** Stable Audio 2.5 和 AudioCraft 2 采用单次前向传播生成固定长度的音频片段。若要实现流式传输，必须对片段进行分块并处理边界重叠（类似于滑动窗口扩散（Sliding-window Diffusion）机制），这会相比编解码器自回归（Codec AR）模型额外增加 100-300ms 的延迟开销。

若产品定位为“实时语音对话”或“实时音乐续写”，应优先选择编解码器自回归（Codec AR）方案；若产品需求是“用户提交后生成一段 30 秒的音频”，则流匹配模型在音质和整体延迟上更具优势。

## 延伸阅读

- [Défossez 等人 (2022). Encodec: 高保真神经音频压缩](https://arxiv.org/abs/2210.13438) —— 编解码器领域的标准之作。
- [Zeghidour 等人 (2021). SoundStream](https://arxiv.org/abs/2107.03312) —— 首个被广泛采用的神经音频编解码器。
- [Kumar 等人 (2023). 基于改进 RVQGAN 的高保真音频压缩 (DAC)](https://arxiv.org/abs/2306.06546) —— DAC 模型。
- [Wang 等人 (2023). 神经编解码语言模型即零样本文本到语音合成器 (VALL-E)](https://arxiv.org/abs/2301.02111) —— VALL-E 模型。
- [Copet 等人 (2023). 简单可控的音乐生成 (MusicGen)](https://arxiv.org/abs/2306.05284) —— MusicGen 模型。
- [Liu 等人 (2023). AudioLDM 2：基于自监督预训练的全局音频生成学习](https://arxiv.org/abs/2308.05734) —— AudioLDM 2 模型。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) —— 采用流匹配技术的 2025 版文本生成音乐模型。