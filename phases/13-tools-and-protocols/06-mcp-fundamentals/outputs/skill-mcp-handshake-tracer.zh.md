---
name: mcp-handshake-tracer
description: 给定类似 pcap 格式的模型上下文协议（Model Context Protocol）客户端-服务器对话记录，为每条消息标注其原语（primitive）、生命周期阶段（lifecycle phase）及能力依赖（capability dependency）。
version: 1.0.0
phase: 13
lesson: 06
tags: [模型上下文协议, json-rpc, 生命周期, 能力]
---

给定从模型上下文协议（Model Context Protocol）会话中捕获的 JSON-RPC 2.0 信封（envelope）序列，生成一份逐步解析说明，明确指出每条消息的原语（primitive）、生命周期阶段（lifecycle phase）以及底层能力标志（capability flag）。

输出内容：

1. 逐条消息标注。针对每条 `{request, response, notification}`，说明：传输方向（客户端至服务器或服务器至客户端）、原语（tools / resources / prompts / roots / sampling / elicitation / lifecycle）、生命周期阶段，以及为使该消息有效而必须协商的能力标志。
2. 能力检查。从记录中重建 `initialize` 交互过程，并列出所有已协商的能力。标记任何因缺少对应能力而违规的消息。
3. 错误诊断。针对每个 JSON-RPC 错误，指出错误代码，并结合上下文分析最可能的原因。
4. 完整性审计。标记缺失以下任一环节的记录：`initialize`、`initialized` 通知、至少一次 `tools/list` 或等效调用、优雅关闭（graceful shutdown）。
5. 规范合规性检查。对照 2025-11-25 版规范的最小字段集，检查每个请求的 `params`。标记缺失的字段。

硬性拒绝条件：
- 任何使用了规范允许集合之外的方法，且未添加 `x-` 前缀的消息。
- 在客户端未声明 `sampling` 能力时发送的任何 `sampling/createMessage` 消息。
- 在收到 `notifications/initialized` 之前发起的任何调用。

拒绝规则：
- 若要求审计非 MCP 协议的记录，请予以拒绝，并指引参考 A2A 规范（第 13 阶段 · 第 19 课）作为替代方案。
- 若要求“修复”记录，请予以拒绝。本技能仅用于标注，不负责重写。修正操作应通过底层实现的 SDK 进行。

输出格式：按消息到达顺序，每条消息输出一行标注：`[phase/primitive/capability] <method or result shape>`。最后附上三行总结，列出所有能力违规项及缺失的生命周期步骤。