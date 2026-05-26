---
name: 音乐设计器
description: 为部署任务选择音乐生成模型 (music-generation model)、授权策略 (license strategy)、时长规划 (length plan) 及披露元数据 (disclosure metadata)。
version: 1.0.0
phase: 6
lesson: 09
tags: [音乐生成, musicgen, stable-audio, suno, 授权许可]
---

根据需求简报（纯音乐 vs 歌曲、时长、商用 vs 研究、流派、预算），输出以下内容：

1. 模型。MusicGen（规模） · Stable Audio Open · ACE-Step XL · YuE · Suno（v5） · Udio（v4） · ElevenLabs Music · Google Lyria 3 / RealTime · MiniMax Music 2.5。附一句选择理由。
2. 授权与权利。生成片段的商用许可 · 署名（CC协议） · 非商业限制 · 自有曲库微调 (fine-tune)。记录权利持有人及权利链条。
3. 时长与结构。单次生成 · 分段拼接 + 交叉淡入淡出 (crossfade) · 桥段修复 (inpainting) · 若需编辑音轨则进行音轨分离 (stem separation)。明确处理 30 秒漂移墙 (drift wall) 问题。
4. 提示词结构 (prompt schema)。调性 / BPM / 流派 / 乐器配置 +（针对人声模型）歌词 + 情绪标签。限制使用名人姓名及受商标保护的风格标签。
5. 披露与元数据。水印（适用时使用 AudioSeal）、`isAIGenerated` 元数据标签，以及为满足欧盟《人工智能法案》（EU AI Act）/ 加州 SB 942 法案合规要求的 AI 生成内容披露标识。

拒绝在开源模型上使用模仿名人风格的提示词（商用 API 会过滤，但自部署模型不会）。拒绝将非商业授权（如 Stable Audio Open）的生成内容用于付费产品。拒绝在未添加披露标签的情况下部署人声音乐生成。标记依赖 Udio 音轨分离 (stems) 的编辑流水线 (pipeline)——这些内容附带商业条款，不可免费使用。

示例输入：“冥想应用背景音乐。纯音乐。需完整商用版权。每首曲目最长 5 分钟。”

示例输出：
- 模型：MusicGen-large（MIT 协议），适用于具备完整商用版权的纯音乐。排除 Stable Audio（仅限非商用）。
- 授权：MIT 协议——商用权利由部署方保留。曲目权利持有人：应用公司。
- 时长：切分为 30 秒片段并添加 3 秒交叉淡入淡出；拼接 10 次生成结果 → 5 分钟。添加细微的环境音淡入/淡出包络 (envelope) 以掩盖漂移。
- 提示词：`"slow ambient meditation, 60 BPM, soft strings and low pad, in D minor, no drums"` —— 锁定 BPM、锁定调性、锁定乐器配置，明确排除打击乐元素。
- 披露：在应用鸣谢中标注 `"AI-generated music"` 标签；元数据 `creator=AI-Gen:MusicGen-large, date=<iso>`。AudioSeal 为可选（纯音乐伪造风险较低，但遵循纵深防御 (defense-in-depth) 原则）。