---
name: 音频加载器
description: 根据目标模型的预期验证原始音频文件，并安全地进行重采样。
version: 1.0.0
phase: 6
lesson: 01
tags: [音频, 语音, 预处理]
---

给定一个音频文件（包含路径、通道数、采样率 (sample rate)、位深度 (bit depth) 和编解码器 (codec)）以及一个目标模型（指定了所需采样率和通道数的自动语音识别 (ASR) / 文本转语音 (TTS) / 分类器 (classifier)），请输出以下内容：

1. 不匹配项 (Mismatches)。列出文件与目标要求不符的所有维度（采样率 (sr)、通道数、时长下限 (duration floor)、削波检查 (clipping check)）。
2. 重采样方案 (Resample plan)。源采样率、目标采样率、重采样库（`torchaudio.transforms.Resample` 或 `librosa.resample`）、抗混叠滤波器 (anti-aliasing filter) 类型。
3. 通道处理方案 (Channel plan)。单声道折叠策略 (Mono fold strategy)（取均值 vs 仅保留左声道），或在模型支持时采用多声道直通 (multichannel pass-through)。
4. 归一化 (Normalization)。峰值归一化 (Peak normalization) 与均方根归一化 (RMS normalization) 的对比、目标 dBFS 值、削波保护 (clipping guard)。
5. 验证代码片段 (Validation snippet)。用于加载文件、执行变换并断言最终数组符合 `(target_sr, dtype, channel_count, range)` 的 Python 代码。

若未使用抗混叠滤波器 (anti-aliasing filter)，则拒绝执行降采样 (downsample)。若未使用重建滤波器 (reconstruction filter)，则拒绝执行超过 2 倍的升采样 (upsample)。对任何削波峰值超过 ±0.999 或直流偏移 (DC offset) 高于 ±0.01 的输入文件进行标记。