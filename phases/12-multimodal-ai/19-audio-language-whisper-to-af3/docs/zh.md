# 音频-语言模型（Audio-Language Models）：从 Whisper 到 Audio Flamingo 3 的演进脉络

> Whisper（Radford 等人，2022 年 12 月）奠定了语音识别（Speech Recognition）的基础——凭借 68 万小时的弱监督多语言语音数据、简洁的编码器-解码器 Transformer（Encoder-Decoder Transformer）架构，它成为了后续所有自动语音识别（ASR）模型发布时必引的基准。但识别不等于推理。要回答“这段录音里有哪些乐器”、“说话人表达了什么情绪”或“第 3 分钟发生了什么”等问题，需要的是音频理解（Audio Understanding）能力，而非单纯的文本转录。Qwen-Audio、SALMONN、LTU 以及 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步构建了这一技术栈：保留 Whisper 级别的编码器，接入 Q-former，使用音频-文本指令数据进行训练，并引入思维链（Chain-of-Thought）推理。本节课程将梳理这一演进脉络。

**类型：** 构建实践
**编程语言：** Python（标准库，对数梅尔频谱图（Log-Mel Spectrogram） + 音频 Q-former 骨架代码）
**前置知识：** 第 6 阶段（语音与音频），第 12 阶段 · 03（Q-Former）
**预计耗时：** 约 180 分钟

## 学习目标

- 从波形计算对数梅尔频谱图（Log-Mel Spectrogram）：包括加窗、快速傅里叶变换（FFT）、滤波器组（Filter Banks）及对数变换。
- 对比编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合架构。分析各自的优势场景。
- 构建音频 Q-former：使用 N 个可学习查询向量（Learnable Queries）与频谱图块（Spectrogram Patches）进行交叉注意力（Cross-Attention）计算。
- 解释级联式（Cascade）与端到端音频-大语言模型（End-to-End Audio-LLM）训练的差异：为何端到端架构在推理任务上更具扩展性。

## 问题背景

Whisper 已经解决了语音识别问题。音频的“光学字符识别（OCR）”式处理已成为一项标准化能力。但“标准化”的边界仅限于文本转录。如果模型无法对其听到的内容进行推理——包括时间定位、说话人区分、情绪识别、音乐结构分析、环境音理解等——仅靠转录无法支撑实际的产品功能。

目前主要有三条清晰的技术路线：

1. 级联架构（Cascade）：Whisper 负责转录，大语言模型（LLM）基于转录文本进行推理。适用于纯语音场景。但在音乐、环境音、多人语音重叠及情绪识别等任务上表现不佳。
2. 端到端音频-大语言模型（End-to-End Audio-LLM）：音频编码器将音频令牌（Audio Tokens）直接输入大语言模型，跳过转录步骤。保留了声学信息（情绪、说话人特征、环境背景）。但需要全新的训练数据。
3. 混合架构（Hybrid）：音频编码器结合文本解码器，同时具备转录与推理能力。Qwen-Audio 和 Audio Flamingo 选择了这一路线。

## 核心概念

### 对数梅尔频谱图 (Log-Mel spectrogram)：输入特征

每个音频编码器 (audio encoder) 都从相同的特征开始：对数梅尔频谱图。

1. 重采样至 16 kHz。
2. 使用 25 毫秒窗口和 10 毫秒步长进行短时傅里叶变换 (Short-Time Fourier Transform)。
3. 取快速傅里叶变换 (FFT) 结果的幅度。
4. 应用梅尔滤波器组 (Mel filter banks，通常为 80 个在 0-8000 Hz 范围内对数间隔的滤波器)，将频率映射至感知频率。
5. 进行对数压缩 (log(1 + x)) 以压缩动态范围。

结果：一个形状为 (T, 80) 的二维数组，其中 T 为时间帧数。对于帧率为 100 Hz 的 30 秒音频片段，形状为 (3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层的类视觉变换器 (ViT-style transformer)，它将对数梅尔频谱图作为时间帧序列进行处理。输出：每个时间帧对应一个隐藏状态向量 (hidden-state vector)。

对于自动语音识别 (Automatic Speech Recognition, ASR)，Whisper 的解码器是一个交叉注意力 (cross-attention) 变换器，它根据编码器输出生成文本词元 (text tokens)。这是标准的编码器-解码器 (encoder-decoder) 架构。

对于音频大语言模型 (Audio Large Language Models, ALMs)，你需要将编码器输出作为另一个大语言模型 (LLM) 的输入。常见模式为：冻结 Whisper 编码器，训练 Q-former，冻结或微调 LLM。

### BEATs 与专用音频编码器

Whisper 主要在以语音为主的数据集上训练。它在处理音乐和环境音频时表现较弱。

BEATs（Chen 等人，2022）是一个在 AudioSet 上训练的自监督 (self-supervised) 变换器。在相同参数量下，它对音乐和环境声音的捕捉能力优于 Whisper。

AF-Whisper（Audio Flamingo 3 的混合架构）：将 Whisper 与 BEATs 的特征拼接 (concat) 作为音频输入。Whisper 负责携带语言信号，BEATs 负责携带声学信号。

### 音频 Q-former

采用与 BLIP-2 视觉 Q-former 相同的模式。固定数量的可学习查询向量 (learnable queries，通常为 32 或 64 个) 在音频编码器的输出帧上进行交叉注意力计算。这些查询向量最终转化为供 LLM 消费的音频词元 (audio tokens)。

训练对齐阶段：仅训练 Q-former，在音频-文本对（如 AudioCaps、Clotho）上使用对比损失 (contrastive loss) 与描述生成损失 (captioning loss)。指令微调阶段：端到端训练，解冻 LLM，在指令数据上进行训练。

### 发展脉络 —— SALMONN、Qwen-Audio 与 AF3

SALMONN（Tang 等人，2023）：Whisper + BEATs + Q-former + LLaMA。首个具备较强推理能力的开源音频大语言模型。在 MMAU 基准测试中的综合得分约为 0.55。

Qwen-Audio（Chu 等人，2023）：架构相似，但在更丰富的数据集上训练，并针对多轮对话进行了优化。MMAU 得分约为 0.60。

LTU —— Listen, Think, Understand（Gong 等人，2023）：使用显式推理数据，专注于音频片段上的思维链 (chain-of-thought)。模型规模较小但目标更聚焦。

Audio Flamingo 3（Goel 等人，2025 年 7 月）：当前的开源最先进模型 (State-of-the-Art, SOTA)。采用 8B 参数的大语言模型主干（Qwen2 7B），拼接 Whisper-large 编码器与 BEATs 特征，使用 64 个查询向量的 Q-former，并在超过 100 万对音频-文本指令数据上进行训练。MMAU 得分达 0.72，在部分子任务上已媲美闭源前沿模型。

AF3 还引入了按需音频思维链机制：模型可以在输出最终答案前，选择性地生成思考词元 (thinking tokens，例如“让我先识别一下乐器：……”）。启用思考机制后，复杂推理任务的准确率可提升 3-5 个百分点。

### 级联架构与端到端架构

级联流水线 (Cascaded pipeline)：

1. Whisper 将音频转录为文本。
2. LLM 基于文本进行推理。

非常适合“总结这期播客”这类任务。但在以下场景会失效：
- “这首歌的情绪是什么？”——情绪蕴含在声音中，而非文字里。
- “说话的是 Alice 还是 Bob？”——需要说话人识别 (speaker identification)。
- “爆炸发生在第几秒？”——文本会丢失时间定位 (temporal grounding) 信息。
- “这是真实音频还是生成音频？”——深度伪造检测 (deepfake detection) 需要依赖声学特征。

端到端架构保留了完整的声学信号。Qwen-Audio 和 AF3 能够原生处理音乐、环境音和情感。

### 2026 年生产实践指南

针对全新的音频理解产品：

- 选择级联架构：如果目标是语音转录，且不涉及音乐或情感推断。
- 选择 AF3 / Qwen-Audio 系列：如果涉及音乐、情感、多说话人场景或复杂的音频推理。

级联架构成本更低、实现更简单。端到端架构能力更强。

### MMAU —— 音频推理基准测试

MMAU（Massive Multimodal Audio Understanding，大规模多模态音频理解）是 2024-2025 年的音频推理基准测试：

- 涵盖语音、音乐和环境音的 10,000 对音频-文本问答对。
- 覆盖分类、时间推理、因果推理和开放式问答。
- 专门测试级联流水线系统性缺失的能力。

开源最先进模型（AF3）得分为 0.72；闭源前沿模型约为 0.78（如 Gemini 2.5 Pro、Claude Opus 4.7）。该差距小于 VideoMME 基准中开源与闭源模型的差距，表明音频大语言模型正日趋成熟。

## 实践应用

`code/main.py`:

- 在标准库中实现 log-Mel 频谱图（log-Mel spectrogram）计算：加窗（windowing）、朴素离散傅里叶变换（naive DFT）与 Mel 滤波器组（Mel filter-bank）。
- 音频 Q-former 骨架代码：基于编码器输出帧，计算查询（Q）、键（K）、值（V）与注意力机制（attention），并生成 N 个词元（token）。
- 在玩具任务上对比级联式（cascaded）与端到端（end-to-end）架构。

## 交付上线

本课时将生成 `outputs/skill-audio-llm-pipeline-picker.md`。针对给定的音频任务（如语音转写、音乐标签分类、情感推断、多说话人日志（speaker diarization）或环境分类），该模块将自动选择级联式、端到端 AF3 或混合架构（hybrid）。

## 练习

1. 计算一段 30 秒音频在 16kHz 采样率、25ms 窗长、10ms 步长（hop）、80 个 Mel 频带下的 log-Mel 频谱图维度。若采样率改为 48kHz，维度将如何变化？

2. 为什么 Whisper 在音乐任务上表现不佳？BEATs 捕获了哪些 Whisper 未能捕获的音频特征？

3. 音频 Q-former 使用 64 个查询向量（queries）与 32 个相比：在何种任务复杂度下 64 个能带来收益？32 个又能在哪些方面节省计算资源？

4. 阅读 AF3 第 4 节关于按需思考（on-demand thinking）的内容。提出三个最能从思维链（chain-of-thought）中受益的音频任务。

5. 使用 AF3 的输出实现一个最简说话人日志（diarization）流水线。你如何标记说话人切换？

## 关键术语

| 术语 | 通俗叫法 | 实际含义 |
|------|-----------------|------------------------|
| Log-Mel 频谱图（Log-Mel spectrogram） | “Mel 特征” | 经过 Mel 滤波器组处理后，由对数幅度值构成的二维（时间、频率）数组 |
| 音频 Q-former（Audio Q-former） | “音频感知器” | 从音频编码器输出到固定长度查询向量的交叉注意力（cross-attention）瓶颈结构，用于向大语言模型（LLM）输入信息 |
| 级联式（Cascaded） | “先 ASR 后 LLM” | 流水线架构：由自动语音识别（ASR）模型进行转写，再由文本大语言模型进行推理；会丢失声学信息 |
| 端到端（End-to-end） | “音频大语言模型” | 音频特征通过 Q-former 直接输入大语言模型；保留完整的声学信号 |
| BEATs | “AudioSet 音频编码器” | 在 AudioSet 数据集上训练的自监督学习（self-supervised learning）Transformer 模型；在音乐与环境声音任务上表现优异 |
| MMAU | “音频推理基准” | 涵盖语音、音乐、环境的 1 万组问答对；2024 年评估标准 |
| 按需思考（On-demand thinking） | “音频思维链” | 模型可在输出最终答案前选择性生成推理词元（tokens），使准确率提升 3-5 个百分点 |

## 延伸阅读

- [Radford 等人 — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu 等人 — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel 等人 — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang 等人 — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong 等人 — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)