---
name: 序列到序列选择器
description: 为新的序列到序列（sequence-to-sequence）任务选择编码器-解码器（encoder-decoder）架构还是仅解码器（decoder-only）架构。
version: 1.0.0
phase: 7
lesson: 8
tags: [transformers, t5, bart, seq2seq]
---

给定一个序列到序列（seq2seq）任务（翻译 / 摘要 / 语音转文本 / 结构化提取 / 重写）、输入与输出长度分布，以及质量与延迟的优先级权衡，请输出以下内容：

1. 架构（Architecture）。从以下选项中选择其一：编码器-解码器（encoder-decoder，如 T5 / BART / Whisper 风格）、仅解码器（decoder-only）指令微调（instruction-tuned）模型、或仅编码器（encoder-only）加提示词模板（prompt template）。需附一句理由。
2. 预训练目标（Pretraining objective）。跨度破坏（span corruption，T5）、去噪（denoising，BART）、下一词预测（next-token，decoder-only），或“跳过预训练，直接微调现有检查点（checkpoint）”。需指明检查点名称。
3. 输入格式化（Input formatting）。任务前缀字符串（task prefix string，T5 风格）对比系统提示词（system prompt，decoder-only）对比原始词元（raw tokens，BART）。需包含对起始符（BOS）与结束符（EOS）的处理说明。
4. 解码策略（Decoding strategy）。集束搜索（beam search）宽度与长度惩罚（length penalty，适用于翻译/摘要），或核采样（nucleus sampling）/ min-p（适用于聊天类任务）。需明确说明针对该任务所选的策略。
5. 评估（Evaluation）。任务适配的评估指标：BLEU / ROUGE / WER / F1 / 精确匹配（exact match）。需包含测试集划分（test split）规模。

拒绝为生成式输出（generative outputs）推荐仅编码器（encoder-only）架构。当输入本身即为对话时，拒绝推荐编码器-解码器（encoder-decoder）架构——仅解码器（decoder-only）架构天然契合对话记忆（conversation memory）机制。若针对语音转文本任务选择仅解码器（decoder-only）架构却未提及将 Whisper 作为需超越的基线（baseline），则需予以标记警告。