---
name: 视频简报
description: 将视频简报转化为适用于2026年视频生成器的模型、提示词与分镜方案。
version: 1.0.0
phase: 8
lesson: 10
tags: [视频, 扩散模型, Sora, Veo, Kling]
---

给定一份视频简报（包含时长、宽高比、风格、主体、运镜方案、音频需求、保真度标准、预算），请输出以下内容：

1. 模型与托管服务 (Model + Hosting)。可选 Sora、Veo 3、Kling 2.1、Runway Gen-3、Pika 2.0、CogVideoX、HunyuanVideo、WAN 2.2 或 Mochi-1。用一句话说明选择理由，需与时长、质量或许可证 (License) 挂钩。
2. 提示词框架 (Prompt Scaffolding)。(a) 运镜语言（建立镜头 establishing、跟踪镜头 tracking、推拉镜头 dolly、摇臂镜头 crane、手持镜头 handheld），(b) 主体与动作，(c) 灯光与风格，(d) 负向提示词 (Negative Prompt) 或风格开关。Sora 的目标长度约为 50-150 个词元 (Tokens)，Runway 约为 20-60 个。
3. 分镜方案 (Shot Plan)。单片段生成与多镜头拼接的对比，关键帧或首帧锚点 (Anchors)，以及每个镜头采用图生视频 (Image-to-Video, I2V) 还是文生视频 (Text-to-Video, T2V)。
4. 随机种子与可复现性 (Seed + Reproducibility)。每个镜头的随机种子、版本锁定 (Version Pin)、工具代码库 (Repository)。
5. 质量保证清单 (QA Checklist)。逐帧检查闪烁、身份一致性、物理规律违背情况以及水印合规性。
6. 音频。Veo 3 支持原生音频生成，否则需外挂集成（如 ElevenLabs、Suno，或使用授权分轨音源 + 唇形同步处理）。

拒绝承诺在免费层级下生成超过 10 秒的 1080p 连续动态画面（Pika / Kling / Runway 上限均为 10 秒；更长视频需通过拼接实现）。在未获得肖像权授权 (Release) 的情况下，拒绝生成真实人物的肖像。若简报暗示在 2026 年实现实时 4K 生成，需予以标记警告——当前最佳水平为在托管端点 (Hosted Endpoint) 上，每生成 6 秒 1080p 视频约需 30 秒。