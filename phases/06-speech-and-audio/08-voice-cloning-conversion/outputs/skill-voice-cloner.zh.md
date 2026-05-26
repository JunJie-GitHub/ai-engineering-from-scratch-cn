---
name: 语音克隆器
description: 为语音克隆部署选择克隆方法（零样本/转换/适配）、同意凭证、水印和安全过滤器。
version: 1.0.0
phase: 6
lesson: 08
tags: [语音克隆, 语音转换, 水印, 同意授权, 安全]
---

根据任务要求（语言、可用参考音频长度、适配预算、许可证限制、同意状态、部署规模），输出以下内容：

1. 方法选择。零样本克隆（Zero-shot clone）(F5-TTS / VibeVoice / Orpheus / OpenVoice V2) · 语音转换（Voice conversion）(kNN-VC / OpenVoice V2 音色) · 说话人适配（Speaker adaptation）(XTTS v2 + LoRA / VITS 全量微调)。
2. 参考音频准备。所需长度、信噪比（Signal-to-Noise Ratio, SNR）(≥ 20 dB)、单声道 16 kHz 以上、静音裁剪、`ref_text`（F5-TTS 必须完全匹配）。拒绝带有背景音乐垫（Music-bed）的参考音频。
3. 同意凭证（Consent artifact）。来自声音所有者的明确录音同意。模板：姓名 + 日期 + 用途 + 范围 + 撤销流程。保存期限 7 年以上。
4. 水印（Watermark）。在每个输出音频中嵌入 AudioSeal 的 16 位有效载荷（Payload）。在持续集成（Continuous Integration, CI）流水线中配置检测器，在发布音频前验证水印是否存在。
5. 安全过滤器（Safety filters）。针对命名实体（名人/政客/未成年人）的提示词拒绝机制；按用户每小时限流；记录每次克隆生成的审计日志；紧急停止开关（Kill-switch）。

拒绝在未制定水印策略的情况下交付克隆功能。无论是否声称已获同意，均拒绝克隆具名名人/政客/未成年人。拒绝长度不足 3 秒或信噪比（SNR）低于 20 dB 的参考音频。拒绝将 F5-TTS 用于商业部署（因其采用 CC-BY-NC 许可证）。在未明确标注口音迁移差异（Accent-transfer gap）的情况下，拒绝跨语言克隆。

示例输入：“无障碍应用：让肌萎缩侧索硬化症（Amyotrophic Lateral Sclerosis, ALS）患者在尚能说话时录入声音，以便失声后通过文本转语音（Text-to-Speech, TTS）系统发声。语言：美式英语。”

示例输出：
- 方法：OpenVoice V2（MIT 许可证，零样本，需 6 秒参考音频）。无障碍用例自带同意授权；患者即为声音所有者。
- 参考音频准备：在录音棚级环境（安静房间、USB 麦克风、24 kHz 采样率）下录制 5 段 6 秒的音频片段。保存原始音频及转录文本。构建质心参考音频（Centroid reference）以提升稳定性。
- 同意授权：数字签名 + 视频确认声明用途（“确诊后声音复用”），存储于加密卷中并保留 10 年。提供撤销热线。
- 水印：AudioSeal 16 位有效载荷编码 `patient_id` + `clip_id`；检测器在 CI 中对每次生成运行验证。
- 安全：硬性过滤命名实体提示词；记录每次生成日志；作用范围（ROI）限制在患者已登录的应用实例内。不暴露 API。