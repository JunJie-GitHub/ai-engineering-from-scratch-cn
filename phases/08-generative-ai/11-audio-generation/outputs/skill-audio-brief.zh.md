---
name: 音频简报
description: 将音频简报转化为涵盖语音合成（Text-to-Speech, TTS）、音乐生成与音效（Sound Effects, SFX）的模型、提示词及评估方案。
version: 1.0.0
phase: 8
lesson: 11
tags: [音频, 语音合成, 音乐, 音效, 音频编解码器]
---

给定一份音频简报（任务类型：语音合成（TTS）/ 音乐生成 / 音效（SFX）/ 声音克隆（Voice Clone），时长，风格，音色或流派，版权许可限制，实时或离线处理，质量基准（Quality Bar）），请输出以下内容：

1. 模型与托管服务（Hosting）。ElevenLabs V3、OpenAI TTS、XTTS v2、Suno v4、Udio、Stable Audio 2.5、MusicGen 3.3B、AudioCraft 2 或 GPT-4o 实时版。附一句选择理由。
2. 提示词格式。语音合成（TTS）：文本 + 音色提示词（3-10 秒样本或音色 ID）+ 情绪/语速标签。音乐生成：流派 + 乐器配置 + 氛围 + 节拍（BPM）+ 结构标记。音效（SFX）：拟声词 + 声源 + 时长提示。
3. 编解码器（Codec）+ 生成器 + 声码器（Vocoder）链路。指明具体的编解码器（如 Encodec 32 kHz、DAC 44 kHz 或自定义方案）及生成器选型（自回归令牌 token-AR 与流匹配 flow-matching 的对比）。
4. 随机种子与可复现性。固定随机种子（Seed Pin）、固定版本（Version Pin）、提示词哈希（Prompt Hash）。
5. 评估指标。语音合成（TTS）采用平均意见得分（Mean Opinion Score, MOS）或 A/B 测试，音乐生成采用 CLAP 评分，TTS 转录采用字符错误率（Character Error Rate, CER），音效（SFX）采用用户听感测试。
6. 安全护栏（Guardrails）。声音克隆需获得授权同意书 + 添加水印（PerTh / SynthID-audio），音乐输出需进行版权扫描，训练数据需进行合规政策检查。

若未获得所有者明确验证的授权同意，一律拒绝进行声音克隆（磁带时代的“3秒提示音”不构成有效授权）。若音乐生成使用了未获许可的参考素材，一律拒绝交付。对于任何目标延迟 < 200 ms 且未采用流式自回归令牌模型（Streaming Token-AR Model）的实时任务，需予以标记警告——基于扩散模型的音频生成（Diffusion-based Audio）在 2026 年仍无法满足首字节时间（Time To First Byte, TTFB）低于 300 ms 的要求。