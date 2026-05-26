---
name: AI SRE 部署计划
description: 为团队设计 AI SRE（Site Reliability Engineering）部署方案——涵盖多智能体（Multi-agent）分诊架构、结构化运行手册（Runbook）、对抗性评估（Adversarial Evaluation）、有限范围的自动修复（Auto-remediation）以及预测性检测（Predictive Detection）策略。
version: 1.0.0
phase: 17
lesson: 23
tags: [AI SRE, 多智能体, 运行手册, 自动修复, 对抗性评估, Datadog Bits AI, NeuBird, 预测性]
---

结合团队规模、事件数量、可观测性（Observability）成熟度及风险承受能力，制定一份 AI SRE 计划。

输出内容：

1. 架构设计。采用多智能体（Multi-agent）架构：主管智能体（Supervisor）+ 日志智能体（Log Agent）+ 指标智能体（Metric Agent）+ 运行手册智能体（Runbook Agent）+ 人工审批关卡（Human Gate）。将专用智能体与现有数据源（Datadog、Grafana、Loki、Confluence）进行对接。
2. 运行手册（Runbook）改造。从非结构化的 Confluence 文档迁移至结构化的 Markdown 格式，包含症状（Symptom）/ 假设（Hypothesis）/ 验证（Verify）/ 执行（Act）等模块。使用 Git 进行版本控制。
3. 产品选型。可选方案包括 Datadog Bits AI、Azure SRE Agent、NeuBird Hawkeye、Incident.io Autopilot 或自研（DIY）。
4. 自动修复（Auto-remediation）范围。限定安全操作集（如重启 Pod、回滚部署、在阈值内扩缩容）。明确禁止清单（如拓扑变更、代码修改、IAM 权限调整、数据库操作）。采用策略即代码（Policy as Code）进行管理。
5. 对抗性评估（Adversarial Evaluation）。为自动修复设置双模型一致性校验关卡（Two-model Agreement Gate）。若模型意见不一致，则升级至人工处理。
6. 预测性检测（Predictive Detection）策略。若考虑引入预测性检测（参考 MIT 89% 准确率的研究结果），需明确具体的执行策略（Actuation Policy）——如呼叫寻呼机（Pager）、预排空（Pre-drain）或自动扩缩容（Auto-scale）；否则该功能仅停留在仪表盘展示层面。

硬性拒绝项：
- 针对大范围变更的自动修复未设置人工审批关卡。拒绝——必须明确列出安全操作集。
- 将非结构化运行手册作为知识库。拒绝——必须要求使用结构化且带版本控制的 Markdown 文档。
- 采用“部署后无需干预（Set it and forget it）”的表述。拒绝——必须明确界定哪些环节是自动化的，哪些不是。

拒绝规则：
- 若每月事件数量少于 10 起，拒绝全面部署 AI SRE——成本高于收益。建议仅实施结构化运行手册。
- 若团队可观测性（Observability）基础薄弱（日志不可检索、指标稀疏），拒绝——AI SRE 会放大劣质数据的问题。
- 若团队提议将“预测性检测 → 自动修复”作为首个功能，拒绝——需先厘清执行策略（Actuation Policy）相关问题。

输出要求：一份单页计划，涵盖架构设计、运行手册改造方案、产品选型、自动修复范围、对抗性校验关卡及预测性检测策略。末尾附上 12 周部署时间表：第 1-4 周实施结构化运行手册，第 5-8 周部署分诊智能体（Triage Agent），第 9-12 周实现有限范围的自动修复。