---
name: mcp-server-designer
description: 设计并搭建包含工具（tools）、资源（resources）及安全默认配置的 MCP（Model Context Protocol）服务器。
version: 1.0.0
phase: 11
lesson: 14
tags: [大语言模型工程, mcp, 工具使用]
---

给定一个业务领域（内部 API、数据库、文件源）以及将挂载该服务器的主机，请输出以下内容：

1. 原语映射（Primitive map）。明确哪些功能将作为 `tools`（执行操作），哪些作为 `resources`（只读数据），哪些作为 `prompts`（用户调用的模板）。每个原语占一行。
2. 认证方案（Auth plan）。标准输入输出（Stdio，适用于受信任的本地环境）、支持 API 密钥的可流式传输 HTTP（Streamable HTTP），或采用 PKCE 的 OAuth 2.1。请做出选择并说明理由。
3. 模式草案（Schema draft）。为每个工具参数编写 JSON Schema，其中的 `description` 字段需针对大模型的工具选择进行优化（而非面向传统 API 文档）。
4. 破坏性操作列表（Destructive-action list）。列出所有会改变系统状态的工具；必须要求设置 `destructiveHint: true` 并经过人工审批。
5. 测试计划（Test plan）。针对每个工具：一项仅验证模式的契约测试（contract test）、一项通过 MCP 客户端进行的往返测试（round-trip test），以及一项红队提示词注入（prompt-injection）测试用例。

严禁交付未经审批路径即可写入磁盘或调用外部 API 的服务器。单个服务器暴露的工具数量不得超过 20 个；若超出，应按业务领域拆分为多个独立的服务器。