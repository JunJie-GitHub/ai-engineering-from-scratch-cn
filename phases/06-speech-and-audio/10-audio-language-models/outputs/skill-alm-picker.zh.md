---
name: alm-picker
description: 为音频理解任务选择音频-语言模型（Audio-Language Model, ALM）、基准测试子集（Benchmark Subset）、输出模态（Output Modality，文本 vs 语音）及安全护栏（Guardrails）。
version: 1.0.0
phase: 6
lesson: 10
tags: [alm, lalm, qwen-omni, audio-flamingo, gemini-audio, mmau]
---

根据任务类型（语音 / 环境音 / 音乐 / 多音频 / 长音频、输出模态、延迟要求、许可证），输出以下内容：

1. 模型。Qwen2.5-Omni-7B · Qwen3-Omni · SALMONN · Audio Flamingo 3 · AF-Next · LTU · GAMA · Gemini 2.5 Pro (API) · GPT-4o Audio (API)。附一句选择理由。
2. 用于验证的基准测试子集。MMAU-Pro 语音 / 环境音 / 音乐 / 多音频 · LongAudioBench · AudioCaps · ClothoAQA。选择与用户任务相匹配的评估维度。
3. 输出模态。仅文本 · 文本 + 语音（Qwen-Omni, GPT-4o Audio）。如需语音输出，需为额外的语音解码器（Speech Decoder）预留资源。
4. 安全护栏。当模型在多音频任务上的得分低于 30%（接近随机猜测）时，拒绝处理需要多音频对比的提示词。对于超过 10 分钟的输入，在送入大型音频-语言模型（Large Audio-Language Model, LALM）前需先进行说话人分离（Speaker Diarization）。
5. 回退/升级策略。明确任务何时应回退至专用模型——例如使用 Whisper 进行转录（Transcription）、BEATs 进行分类（Classification）、pyannote 进行说话人分离。LALM 并非在所有细分任务上都是最优解。

若未在 MMAU-Pro 多音频子集上验证模型得分超过 40%，则拒绝交付多音频对比任务。若未进行上游说话人分离处理，则拒绝处理长音频（> 10 分钟）。对于任何仅依赖厂商报告数据而未经独立复验的部署，必须标记警告。

示例输入：“合规审计：转录 10 分钟的银行通话录音 + 检测客服是否宣读了强制披露条款。”

示例输出：
- 模型：使用 Whisper-large-v3-turbo 进行转录 + 通过 API 调用 Gemini 2.5 Pro 对转录文本进行披露条款检查问答。直接在原始音频上使用 LALM 虽具吸引力，但长音频 LALM 的准确率在超过 10 分钟后会显著下降。
- 基准测试子集：MMAU-Pro 语音子集（Gemini 2.5 Pro = 73.4%）——覆盖语音推理维度。同时使用自有的 50 通通话黄金数据集进行抽样检查。
- 输出模态：仅文本。审计报告无需语音输出。
- 安全护栏：首先使用 pyannote 3.1 进行说话人分离；按说话人分段单独输入；记录每通通话的置信度分数。
- 回退/升级策略：若某通通话未通过披露检查，应转交人工审核，而非自动标记。