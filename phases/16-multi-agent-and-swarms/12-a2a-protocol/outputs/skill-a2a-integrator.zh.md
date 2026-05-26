---
name: a2a-integrator
description: 设计两个智能体（Agent）之间的 A2A（Agent-to-Agent）集成方案——包括智能体卡片（Agent Card）、任务模式（Task Schemas）、认证（Auth）、流式传输（Streaming）或轮询（Polling）。
version: 1.0.0
phase: 16
lesson: 12
tags: [多智能体, a2a, 协议, 互操作性, google]
---

给定两个需要互操作的智能体系统，请制定 A2A 集成计划：涵盖智能体卡片内容、任务模式、认证和传输模式。

输出内容：

1. **智能体卡片（Agent Card）**。名称、版本、技能、端点、支持的模态（Modalities）（文本、结构化数据、图像、音频、视频）、`protocol_version`、认证声明。
2. **每项技能的任务模式（Task Schemas）**。输入 JSON 模式（JSON Schema） + 产出物（Artifact）JSON 模式。务必明确具体——客户端将对其进行验证。
3. **认证（Auth）选择**。Bearer 令牌（OAuth2 或不透明令牌）、mTLS（双向 TLS）或签名请求。请根据威胁模型（Threat Model）（公共互联网、VPC 虚拟私有云或混合环境）说明选择理由。
4. **传输模式**。轮询（Polling）对比 SSE（Server-Sent Events）流式传输对比 Webhook 回调。长耗时或进度反馈密集的任务使用流式传输；短任务使用轮询。
5. **速率限制（Rate Limits）**。按客户端和按任务的限制。防止滥用。
6. **幂等性（Idempotency）**。处理重复 `POST /tasks` 请求的策略（客户端任务键、服务端去重（Deduplication））。
7. **失败处理（Failure Handling）**。除 `failed` 之外的任务状态（可重试与致命错误）、死信策略（Dead-Letter Policy）、错误产出物模式。
8. **MCP 与 A2A 的边界划分**。如果远程智能体内部使用 MCP（Model Context Protocol），请注明哪些工具对外暴露，哪些保留在内部。

硬性拒绝条件：

- 未声明协议版本的智能体卡片。
- 在用例需要结构化数据时，仍使用自由文本的任务模式。
- 在公共互联网部署中设置 `Auth=none`（无认证）。

拒绝规则：

- 如果两个智能体运行在同一进程中，拒绝采用 A2A 方案，并建议直接使用 Python/JS 调用。A2A 适用于跨系统边界场景。
- 如果延迟要求低于 100ms 往返时间，拒绝采用 A2A 方案，并建议使用共享模式的直接 RPC（远程过程调用）。
- 如果远程智能体未声明智能体卡片，拒绝集成，并建议先发布该卡片。

输出：一份单页集成简报。末尾以内联形式粘贴智能体卡片 JSON，以便工程团队直接将其放入 `/.well-known/agent.json`。