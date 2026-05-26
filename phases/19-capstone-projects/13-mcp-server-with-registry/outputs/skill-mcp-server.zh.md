---
name: mcp-server-platform
description: 部署一个生产级模型上下文协议（Model Context Protocol）服务器，支持可流式 HTTP（StreamableHTTP）、OAuth 2.1 作用域（scopes）、开放策略代理（Open Policy Agent）策略、针对破坏性工具的人工审批门控（human-approval gate），以及用于服务发现的注册中心（registry）。
version: 1.0.0
phase: 19
lesson: 13
tags: [综合项目, mcp, fastmcp, streamablehttp, oauth, opa, 注册中心, 治理]
---

在企业环境中，交付一个包含 10 个内部工具的 MCP 服务器、一个用于服务发现的注册中心服务，以及一个通过 Slack 审批来管控破坏性工具的治理层。

构建计划：

1. 使用 FastMCP 服务器暴露 10 个只读工具（Postgres、S3、Jira、Linear、Datadog、PagerDuty、GitHub、Notion、Slack、Salesforce），每个工具均具备类型化模式（typed schema）和必需作用域（required scope）。
2. 采用 StreamableHTTP 传输协议，在负载均衡器后方保持无状态（stateless）运行。
3. OAuth 2.1 令牌自省（token introspection）中间件；通过 SPIFFE / SPIRE 实现工作负载身份（workload identity）验证。
4. 针对每次工具调用执行 OPA / Rego 策略决策：强制作用域检查、个人身份信息（Personally Identifiable Information）脱敏以及载荷大小限制（payload size caps）。
5. 将破坏性工具（Jira 创建、Linear 创建、Postgres 写入）部署在独立的 MCP 服务器上，要求具备 `approved:by:human` 作用域，该权限需在 15 分钟内通过 Slack 卡片审批提升。
6. 注册中心服务定期轮询各服务器的 `.well-known/mcp-capabilities` 端点，使用 JSON Schema 进行验证，并提供列表/搜索/验证/启用的用户界面。
7. 按租户划分的 JSONL 审计日志，在写入前使用 Presidio 进行 PII 脱敏。
8. 进行 100 客户端负载测试以验证水平扩展能力；通过 MCP 一致性测试套件。

评估量规：

| 权重 | 评估标准 | 衡量指标 |
|:-:|---|---|
| 25 | 规范一致性 | StreamableHTTP + 能力清单（capability manifest）通过 MCP 一致性测试 |
| 20 | 安全性 | 作用域强制执行、覆盖所有工具的 OPA 策略、密钥管理规范 |
| 20 | 可观测性 | 每次工具调用的审计日志，写入时进行 PII 脱敏 |
| 20 | 扩展性 | 100 客户端负载测试及水平扩展演示 |
| 15 | 注册中心用户体验 | 完整演练发现/验证/启用-禁用工作流 |

硬性否决项：

- 要求有状态会话的服务器（违反 2026 版 StreamableHTTP 无状态契约）。
- 单服务器拓扑结构，其中破坏性工具与只读工具共享相同的认证面（auth surface）。
- 持久化存储原始 PII 的审计日志。
- 忽略能力清单；注册中心集成是硬性要求。

拒绝规则：

- 未配置 OAuth 则拒绝部署；匿名访问将直接取消资格。
- 未集成 Slack 审批流程则拒绝交付破坏性工具。
- 拒绝暴露作用域或描述未包含在能力清单中的工具。

输出：一个代码仓库，包含两个 MCP 服务器（只读型与破坏型）、注册中心服务、Slack 审批集成、OPA 策略、100 客户端负载测试框架、一致性测试结果，以及一份说明文档。说明文档需阐述你曾考虑暴露但最终未暴露的工具（及原因），并列出在预演（dry-run）期间拦截了三次“险些违规”操作（near-misses）的前三条 OPA 规则。