---
name: rollout-runbook
description: 为新的大语言模型（LLM）或提示词模板（Prompt Template）设计一套从影子模式（Shadow）→ 金丝雀发布（Canary）→ A/B 测试 → 100% 全量发布的发布计划，包含五个金丝雀检查点（Canary Gates）、感知噪声底线的阈值（Noise-Floor-Aware Thresholds），以及秒级快速回滚路径。
version: 1.0.0
phase: 17
lesson: 20
tags: [发布, 金丝雀发布, 影子模式, 渐进式交付, 特性开关, Argo Rollouts, Flagger, KServe]
---

给定候选变更（新模型、新提示词模板、新路由策略）、线上基线指标（Baseline Production Metrics）和风险容忍度，输出一份发布运行手册（Rollout Runbook）。

输出内容：

1. 影子模式计划（Shadow Plan）。持续时间：24-72 小时。记录指标：输出内容、Token 数量、延迟（Latency）、拒绝率、错误率。告警条件：成本波动 >20%，输出长度变化 >30%，或任何模式违规（Schema Violation）。
2. 金丝雀发布推进（Canary Progression）。阶段划分：1% → 10% → 25% → 50% → 75% → 100%。每阶段持续时间：30 分钟至 24 小时（根据流量规模调整；确保每个阶段有足够的数据以达到统计置信度（Statistical Confidence））。
3. 五个检查点（Gates）。明确指定延迟 P99、单次请求成本、错误/拒绝率、输出长度 P99、点踩率（Thumbs-down Rate）的精确阈值。阈值需设置在噪声底线（Noise Floor）之上（预期存在 15% 的不可缩减方差）。
4. 工具链（Tooling）。指定发布控制器（如 Argo Rollouts、Flagger、KServe）以及用于即时回滚的特性开关（Feature Flag）系统。
5. 回滚路径（Rollback Path）。记录三个操作步骤：切换开关 → 回退至固定摘要版本（Revert Pinned Digest）→ 验证。目标耗时：端到端不超过 60 秒。
6. 是否跳过 A/B 测试？需提供理由。优化型变体变更可跳过 A/B 测试；具有显著差异的变更（新行为、新成本曲线）必须进行 A/B 测试。

硬性拒绝条件（Hard Rejects）：
- 跳过影子模式。拒绝——成本激增和长度退化会绕过离线评估（Offline Eval）。
- 检查点阈值严于 15% 方差。拒绝——误报将中断合法的发布流程。
- 需要重新部署的回滚。拒绝——这不叫回滚，这叫事故报告。

拒绝规则（Refusal Rules）：
- 若变更涉及安全关键项（如个人身份信息（PII）处理变更），需增加明确附加检查点：在启动金丝雀发布前，影子样本中必须实现零 PII 泄露。
- 若流量低于 100 请求/小时，需延长金丝雀阶段——否则检查点噪声将淹没有效信号。
- 若团队无法提供五个金丝雀检查点的基线指标，则拒绝发布——基线数据是必要前提。

输出要求：一份单页运行手册，涵盖影子模式、金丝雀发布、检查点、工具链、回滚路径及 A/B 测试策略。末尾需附加回滚演练要求：在首次真实部署前，必须进行一次回滚演练。