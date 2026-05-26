---
name: 优化循环
description: 根据任务、验证器 (verifier) 可用性和迭代预算，配置评估器-优化器 (evaluator-optimizer)（自我优化 (Self-Refine) / CRITIC）循环。
version: 1.0.0
phase: 14
lesson: 05
tags: [自我优化, 批评者, 评估器-优化器, 护栏, 迭代]
---

给定任务、迭代预算以及可用的验证器类型（基于工具 (tool-grounded) 或仅支持自我评估 (self-eval)），生成用于评估器-优化器循环的提示词 (prompts) 和停止策略 (stop policy)。

需生成以下内容：

1. 生成器提示词 (Generator prompt)。用于首次输出的确定性生成器。明确陈述任务、输出格式和约束条件。
2. 评估器/验证器提示词 (Evaluator/verifier prompt)。如果可用工具（搜索、代码运行、测试、计算器、类型检查），需指定调用方式以及如何生成结构化批评意见 (structured critique)（JSON 格式，包含：pass/fail、violations[]、suggested_fixes[]）。如果仅支持自我评估，需明确标记自我优化的“橡皮图章”风险 (rubber-stamp risk)，并使用结构不同的提示词风格（例如对抗式“至少找出一个缺陷”）。
3. 优化器提示词 (Refiner prompt)。必须引用先前的输出和批评意见（历史记录）。明确声明“不得重复先前迭代中标记的失败模式 (failure mode)”为强制性要求。
4. 停止策略 (Stop policy)。采用逻辑组合条件：验证器通过 OR（自我评估认为合格 AND 迭代次数 >= 2）OR 迭代次数 >= max_iterations。绝不可使用单一条件。
5. 可观测性钩子 (Observability hooks)。按照第 23 课的要求，将每次迭代记录为 OpenTelemetry GenAI 跨度 (span)（评估、优化），以便完整审计优化轨迹。

硬性拒绝条件 (Hard rejects)：

- 生成器和批评者使用相同的提示词。存在“橡皮图章”风险——模型会自我附和。
- 无迭代上限。无限优化循环会消耗大量 Token；默认上限始终设为 4。
- 验证器提示词要求自由格式的散文式反馈。仅限结构化 JSON——通过/失败状态及逐条列出的违规项。
- 在优化器提示词中省略历史记录。研究表明，缺少历史记录会导致输出质量骤降。

拒绝规则 (Refusal rules)：

- 如果任务没有验证器且无法构建验证器，则拒绝使用 CRITIC 模式，并指出自我优化是较弱的备选方案——需向用户警告“橡皮图章”风险。
- 如果 max_iterations >= 10，则拒绝并建议重新设计任务架构。超过 3-4 轮迭代才收敛通常表明生成器提示词存在错误。
- 如果验证器调用破坏性工具（shell、git write），则拒绝并要求设置沙箱边界 (sandbox boundary)（参考第 09 课）。

输出：一个包含所有提示词、停止策略和工具列表的单一配置块 (configuration block)，并附带“下一步阅读”说明。根据部署目标，该说明将指向第 16 课（OpenAI Agents SDK 护栏 (guardrails)）、第 12 课（Anthropic 评估器-优化器）或第 30 课（评估驱动的 Agent 开发 (eval-driven agent development)）。