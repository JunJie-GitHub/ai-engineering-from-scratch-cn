---
name: a2a-agent-spec
description: 为支持通过 A2A 调用的智能体生成智能体卡片（Agent Card）与技能模式（skills schema）。
version: 1.0.0
phase: 13
lesson: 18
tags: [a2a, 智能体卡片, 任务生命周期, 委托]
---

根据智能体的能力与预期协作方，生成其 A2A 智能体卡片（Agent Card）与技能定义。

需生成以下内容：

1. 智能体卡片（Agent Card）。包含 `name`、`description`、`url`、`version`、`schemaVersion`、`capabilities`（streaming、pushNotifications）以及 `skills[]`。
2. 技能列表。每项需包含 `id`、`name`、`description`、`inputModes`、`outputModes`。描述部分需采用“适用于 X 场景。不适用于 Y 场景。”的表述模式。
3. 任务状态规划（Task-state plan）。针对每项技能，明确预期的状态流转路径及 `input_required` 路径。
4. 签名方案（Signing plan）。说明是否通过 AP2 对卡片进行签名（建议对外部可调用的智能体采用此方案）。
5. 传输协议（Transport）。采用基于 HTTP 的 JSON-RPC（默认）或 gRPC。需注明与 v1.0 的向后兼容性（backward-compatibility）。

硬性拒绝条件：
- 缺少稳定 URL 的智能体卡片。这将导致服务发现（discovery）失败。
- 未声明输入与输出模式的技能。调用方将无法推断兼容性。
- 对外部可调用但未提供 AP2 签名方案的智能体。存在身份冒充攻击面（impersonation vector）。

拒绝规则：
- 若智能体的用例仅为单次工具调用，则拒绝搭建 A2A 脚手架（scaffold）；建议改用模型上下文协议（MCP）。
- 若智能体暴露了不应公开的内部信息（如工具调用追踪、思维链（chain-of-thought）），则予以拒绝，并强制要求实现信息不透明（opacity）。
- 若智能体需借助 A2A 处理支付（属 AP2 用例），需确认 AP2 扩展版本，并明确标注 AP2 独立于核心 A2A 协议。

输出要求：一份单页的智能体卡片（Agent Card）JSON、每项操作的技能模式（skills schema）、状态流转规划、签名与传输协议选择。末尾需附上该智能体承诺的最低 v1.0 向后兼容性保证。