---
name: whisper-tuner
description: 为指定的语言、领域和延迟预算 (Latency Budget) 设计 Whisper 微调 (Fine-tuning) 或推理流水线 (Inference Pipeline)。
version: 1.0.0
phase: 6
lesson: 05
tags: [音频, whisper, 自动语音识别 (ASR), 微调, 低秩自适应 (LoRA)]
---

给定目标（语言集合、领域、片段长度分布、延迟预算、硬件）和数据（可用时长、质量），输出：

1. 模型变体 (Variant)。Tiny / Base / Small / Medium / Large-v3 / Turbo。选择理由。
2. 运行时 (Runtime)。vanilla / faster-whisper / whisperx / whisper-streaming。选择理由。
3. 微调计划。全量微调 (Full-FT) 与低秩自适应 (LoRA)（秩 `r`、目标模块 `target_modules`）、编码器冻结策略 (Freeze-Encoder Policy)、训练轮数 (Epoch Count)。
4. 推理防护机制 (Inference Guards)。语音活动检测 (Voice Activity Detection, VAD)（Silero 或 Whisper 自带）、`temperature=0`、`condition_on_previous_text=False`、`no_speech_threshold`。
5. 评估 (Evaluation)。领域词错误率 (Word Error Rate, WER) 目标、文本规范化 (Text Normalization) 规则、静音片段幻觉率 (Hallucination Rate) 检查。

拒绝在未使用语音活动检测 (VAD) 的情况下将 Whisper 部署于任意音频。拒绝在缺乏防失控防护机制的情况下，为多片段任务设置 `condition_on_previous_text=True`。标记任何替换了 Whisper 分词器 (Tokenizer) 或梅尔频谱处理流水线 (Mel Pipeline) 的微调方案。