---
name: codec-picker
description: 为指定的生成或压缩任务选择合适的神经音频编解码器（Neural Audio Codec）（如 EnCodec / DAC / SNAC / Mimi）。
version: 1.0.0
phase: 6
lesson: 13
tags: [编解码器, EnCodec, DAC, SNAC, Mimi, 残差矢量量化, 语义词元]
---

根据具体任务（生成式大语言模型 (Generative LM)、音频压缩、全双工对话 (Full-Duplex Dialogue)、音乐编辑、保真度目标 (Fidelity Target)），输出以下内容：

1. 编解码器 (Codec)。EnCodec-24k · EnCodec-48k · DAC-44.1k · SNAC-24k · Mimi ·（备选方案：非神经压缩使用 Opus）。附一句选择理由。
2. 帧率 (Frame Rate) + 码本 (Codebooks)。比特率预算 (Bitrate Budget)、码本数量（通常为 4-12 个）、目标音频片段时长对应的序列长度。
3. 词元化方案 (Tokenization Scheme)。扁平结构 (Flat) vs 层次结构 (Hierarchical)（SNAC）vs 语义+声学结构 (Semantic + Acoustic)（Mimi）。说明大语言模型如何处理词元。
4. 解码器 (Decoder)。编解码器内置解码器 (In-Codec Decoder) · 外部声码器 (External Vocoder)（如 HiFi-GAN）· 仅大语言模型 (LM-Only)（无需声码器，直接预测编解码器词元）。解释选择原因。
5. 训练影响 (Training Implications)。是否需要训练编码器/解码器？是否需在特定领域音频上进行微调 (Fine-Tune)（仅语音 → 特定领域音乐）？是否直接使用冻结的现成模型 (Frozen Off-the-Shelf)？

在延迟预算 (Latency Budget) 严格的自回归大语言模型 (AR-LM) 任务中，拒绝使用 DAC —— 86 Hz 帧率 × 8 个码本 = 每 10 秒产生 5,504 个词元，序列过长不利于快速生成。在音乐任务中，拒绝使用 Mimi —— 该模型针对语音优化。在语义条件生成 (Semantic-Conditional Generation) 任务中，拒绝使用 EnCodec —— 缺乏语义码本，会导致文本生成的语音模糊不清。

示例输入：“构建一个用于文本转语音 (Text-to-Speech, TTS) 的自回归大语言模型 (AR LM)。目标首音频延迟 (Time to First Audio, TTFA) 为 200 毫秒。仅支持英语。”

示例输出：
- 编解码器：Mimi。语义与声学的分离结构支持“文本 → 码本 0 → 码本 1-7”的分解生成策略，既能实现快速生成，又支持声音克隆。
- 帧率 + 码本：12.5 Hz · 8 个码本 · 4.4 kbps。10 秒音频 = 1,000 个词元。
- 词元化：首先根据文本和说话人参考音频预测码本 0；随后在给定码本 0 和说话人参考音频的条件下预测码本 1-7（深度 Transformer 模式 (Depth-Transformer Pattern)）。
- 解码器：使用 Mimi 内置解码器，无需外部声码器。
- 训练：训练文本到编解码器词元的大语言模型；冻结 Mimi 模型参数。