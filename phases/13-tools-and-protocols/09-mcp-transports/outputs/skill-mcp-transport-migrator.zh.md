---
name: mcp-transport-migrator
description: 制定从传统 HTTP+SSE 迁移至可流式传输 HTTP（Streamable HTTP）的迁移计划，需保持会话 ID（Session ID）连续性并进行来源（Origin）验证。
version: 1.0.0
phase: 13
lesson: 09
tags: [mcp, streamable-http, sse-migration, session-id, origin]
---

针对现有的传统 HTTP+SSE（HTTP + Server-Sent Events）MCP（Model Context Protocol）服务器，制定迁移至单端点可流式传输 HTTP（Streamable HTTP）的迁移计划。

需产出以下内容：

1. 端点（Endpoint）重写。将 `/messages` 和 `/sse` 合并为单一的 `/mcp`。将 POST 请求映射至请求处理逻辑，GET 请求映射至 SSE 数据流，DELETE 请求映射至会话终止操作。
2. 会话连续性（Session Continuity）。在首次 POST 请求时生成新的 `Mcp-Session-Id`。拒绝客户端提供的 ID。若客户端首次发送的是传统会话 Cookie，则保留桥接逻辑。
3. 来源验证（Origin Validation）。将明确的生产环境来源（`https://app.company.com`、`https://claude.ai` 及 localhost 变体）加入白名单。对其余所有请求返回 403 状态码予以拒绝。
4. `Last-Event-Id` 重放（Replay）。为每个会话维护一个近期事件的环形缓冲区（Ring Buffer），以便断线重连时能够恢复。
5. 弃用过渡期（Deprecation Window）。明确记录系统切换日期，并设置 60 天的宽限期。在此期间，传统端点将通过 301 状态码重定向至新端点，并附带警告响应头（Warning Header）。

硬性否决条件（Hard Rejects）：
- 任何计划若让新旧端点永久共存，一律拒绝。传统 SSE 将于 2026 年移除。
- 任何由客户端生成会话 ID（Session ID）的计划，一律拒绝。这违反了密码学随机性要求。
- 任何未包含来源验证（Origin Validation）的计划，一律拒绝。存在 DNS 重绑定（DNS Rebinding）漏洞风险。

拒绝执行规则（Refusal Rules）：
- 若服务器仅限本地使用（标准输入/输出，stdio），则拒绝迁移至 HTTP；对于本地场景，stdio 是正确选择。
- 若服务器尚未集成 OAuth，请在公开暴露前完成第 13 阶段 · 第 16 课的内容。
- 若托管目标不支持长连接 HTTP（例如 Vercel 免费版），则拒绝迁移，并建议改用 Cloudflare Workers。

输出内容：一份迁移操作手册（Migration Runbook），需包含端点变更说明、来源白名单、会话 ID 管理方案、弃用时间表，以及一份测试检查清单。该清单需覆盖初始化（initialize）、工具列表查询（tools/list）、流式通知（streaming notifications）、携带 `last-event-id` 的重连测试，以及显式 DELETE 请求测试。