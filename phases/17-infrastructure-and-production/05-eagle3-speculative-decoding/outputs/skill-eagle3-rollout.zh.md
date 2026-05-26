---
name: eagle3-rollout
description: 制定分阶段的 EAGLE-3 推测解码（Speculative Decoding）发布计划，在正式上线前于真实流量中测量接受率 alpha（Acceptance Rate Alpha）。
version: 1.0.0
phase: 17
lesson: 05
tags: [推测解码, eagle-3, vllm, alpha, 生产环境发布]
---

给定目标模型、硬件（GPU 类型与数量）、流量描述（通用对话 / 代码 / 垂直领域）、并发目标以及当前基线指标（首字延迟 TTFT、词元间延迟 ITL、吞吐量 Throughput），制定一份分阶段的 EAGLE-3 发布计划。

输出内容：

1. 基线测量计划。选择何种基准测试工具（LLMPerf、GenAI-Perf 或生产环境影子流量）、提示词分布（Prompt Distribution）、并发测试点，以及需要记录的指标（TTFT 均值/P99、ITL 均值/P99、吞吐量、并发数）。
2. 草稿头（Draft Head）模型选择。通用对话场景使用基于 ShareGPT 训练的 EAGLE-3。垂直领域流量（代码、医疗、法律）使用领域微调的 EAGLE-3，或决定在发布前训练专用模型。
3. 配置。明确 vLLM `speculative_config` 的具体字段（method、model、num_speculative_tokens）。注意 v0.18.0 版本的兼容性限制：草稿模型推测解码无法与 `--enable-chunked-prefill` 同时启用；V1 版本中的 N-gram GPU 推测解码是例外。
4. Alpha 阈值门控。在生产并发级别下，目标 alpha（接受率）需 >= 0.55。测量流程：进行 24 小时影子流量测试，记录 vLLM `spec_decode_metrics`，将接受的 Token 数除以请求的草稿长度。若任意 1 小时窗口内 alpha 降至 0.45 以下，则触发熔断开关（Kill Switch）。
5. 长尾延迟监控。绘制 P99 ITL 差值图（开启推测解码 - 关闭推测解码）。若差值为正，说明被拒绝的草稿导致的双遍处理模式（Two-Pass Pattern）正在产生负面影响。针对该工作负载降低 K 值或禁用推测解码。
6. 盈亏平衡检查。在报告的并发级别下，计算当前验证开销（Verify Overhead）对应的盈亏平衡 alpha 值。仅当实测 alpha 值超出盈亏平衡点至少 0.1 时，方可正式发布。

硬性拒绝条件：
- 未在生产流量中测量 alpha 即发布。予以拒绝，并要求进行 24 小时影子流量测量。
- 声称实现 2-3 倍加速但未提供实测 alpha 值。
- 在延迟非瓶颈的离线批处理任务中启用推测解码。
- 在 vLLM v0.18.0 中将草稿模型推测解码与分块预填充（Chunked Prefill）结合使用。存在硬性不兼容。

拒绝规则：
- 若流量主要为极短输出（平均 Token 数低于 50），予以拒绝。草稿开销将占据主导；应直接发布原始目标模型。
- 若硬件为消费级显卡（RTX 4090 / 5090）且批处理大小（Batch Size）始终低于 8，建议直接发布原始目标模型——验证开销的批处理摊销（Batch Amortization）需要硬件无法提供的高并发支持。
- 若用户希望在无测量循环的情况下自动调优（Auto-Tune）K 值，予以拒绝。K 值需根据实测 alpha 与验证开销综合选定；任何自动调优都无法替代实际测量。

输出要求：一份单页的分阶段发布计划，依次列出基线测量 → 配置 → Alpha 阈值门控 → 长尾延迟监控 → 盈亏平衡确认。最后附上一段“下一步测量建议”，根据诊断结果明确指出是进行领域特定的 EAGLE-3 训练、降低 K 值，还是回退至原始目标模型。