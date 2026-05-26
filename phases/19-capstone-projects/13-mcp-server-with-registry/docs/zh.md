# 综合实战项目 13 — 带注册中心与治理的 MCP 服务器

> 模型上下文协议（Model Context Protocol, MCP）已不再是未来的概念，而是于 2026 年成为默认的工具调用规范。Anthropic、OpenAI、Google 以及所有主流集成开发环境（IDE）均内置了 MCP 客户端。Pinterest 公开了其内部的 MCP 服务器生态系统。AAIF 注册中心（AAIF Registry）在 `.well-known` 路径下规范了能力元数据。AWS ECS 发布了参考级无状态部署方案。Block 的 goose-agent 将同一协议集成到了托管助手之中。2026 年的生产级架构形态为：采用 StreamableHTTP 传输协议、OAuth 2.1 作用域（scopes）、开放策略代理（Open Policy Agent, OPA）策略网关，以及一个允许平台团队发现、验证和启用服务器的注册中心。请端到端地构建该系统。

**类型：** 综合实战项目
**编程语言：** Python（服务器端，通过 FastMCP）或 TypeScript（@modelcontextprotocol/sdk），Go（注册中心服务）
**前置要求：** 第 11 阶段（大语言模型工程）、第 13 阶段（工具与 MCP）、第 14 阶段（智能体）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P11 · P13 · P14 · P17 · P18
**预计耗时：** 25 小时

## 问题背景

MCP 已成为工具调用的事实标准。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及所有托管型智能体（agents）现在均依赖 MCP 服务器。生产环境中的挑战并非开发服务器（FastMCP 已使其变得十分简单），而是如何以企业级要求进行大规模部署：包括按租户隔离的 OAuth 作用域（scopes）、针对破坏性工具的 OPA 策略控制、基于 StreamableHTTP 的无状态水平扩展、用于服务发现的注册中心，以及每次工具调用的审计日志。Pinterest 的内部 MCP 生态系统与 AAIF 注册中心规范共同确立了 2026 年的行业标准。

你将构建一个暴露 10 个内部工具的 MCP 服务器（如 Postgres 只读查询、S3 列表获取、Jira、Linear、Datadog 等），一个供平台团队发现服务的注册中心 UI，以及针对破坏性工具的人工审批网关。负载测试将验证 StreamableHTTP 的水平扩展能力，而完整的审计轨迹将满足企业级安全审查要求。

## 核心概念

MCP 2026 修订版强制要求将 StreamableHTTP 作为默认传输协议。与早期基于标准输入输出（stdio）和服务器发送事件（SSE）的架构不同，StreamableHTTP 默认是无状态的：单个 HTTP 端点接收 JSON-RPC 请求，流式返回响应，并支持用于通知的长连接。无状态特性意味着该服务可置于负载均衡器后方进行水平扩展。

授权机制采用 OAuth 2.1 并支持按工具划分的作用域。访问令牌（token）携带诸如 `jira:read`、`s3:list`、`postgres:query:readonly` 等作用域。MCP 服务器会在每次工具调用时校验作用域，而非仅在会话开始时校验。对于高风险工具，若其作用域在过去 N 分钟内未被提升至 `approved:by:human`，服务器将拒绝任何调用请求——该权限提升操作源自 Slack 审批卡片。

注册中心是一项独立服务。每个 MCP 服务器都会在 `.well-known/mcp-capabilities` 路径下暴露一份文档，其中包含其工具清单、传输 URL 及认证要求。注册中心负责定期轮询、验证并建立索引。平台团队可通过注册中心 UI 查看可用工具、所需作用域以及负责维护的团队。

## 架构设计

MCP 客户端（Claude Code、Cursor 3 等）
          |
          v
基于 HTTPS 的 StreamableHTTP（JSON-RPC + 流式传输）
          |
          v
负载均衡器后的 MCP 服务器（FastMCP）
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 列表    Jira       Linear     Datadog
（只读）    （分页）    （只读）    （只读）    （查询）
          |
   +------+-------------+
   v                    v
 OPA 策略网关        破坏性工具 MCP（独立服务器）
                        |
                        v
                   通过 Slack 进行人工审批
                        |
                        v
                   审计日志（仅追加，按租户隔离）

  注册表服务
     |
     v  从各服务器获取 GET /.well-known/mcp-capabilities
     v
     UI：搜索 / 验证 / 启用-禁用 / 所有权管理

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- 传输协议：基于 HTTPS 的 StreamableHTTP（无状态 Stateless）
- 身份认证：OAuth 2.1，通过 SPIFFE / SPIRE 实现工作负载身份（Workload Identity）
- 策略控制：基于 OPA / Rego 的按工具规则；按请求的策略决策服务（Policy Decision Service）
- 注册表：自托管，消费 `.well-known/mcp-capabilities` 清单文件（Manifests）
- 人工审批：针对破坏性工具（Destructive Tools）使用 Slack 交互式消息
- 部署：AWS ECS Fargate 或 Fly.io，每租户独立服务器或共享服务器（带租户作用域隔离 Tenant Scoping）
- 审计：按租户隔离的结构化 JSONL 存储桶，记录每次调用的血缘关系（Lineage）

## 构建指南

1. **工具暴露面（Tool surface）**。公开 10 个内部工具：Postgres 只读查询、S3 对象列表、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 只读。每个工具均具备类型化模式（typed schema）与作用域标签（scope label）。

2. **FastMCP 服务器（FastMCP server）**。挂载上述工具。配置 StreamableHTTP 传输协议。添加用于 OAuth 令牌内省（OAuth token introspection）与作用域强制（scope enforcement）的中间件。

3. **OPA 策略（OPA policy）**。为每个工具编写 Rego 策略：规定允许调用的作用域、适用的个人身份信息（PII）脱敏规则，以及载荷大小上限（payload-size caps）。每次工具调用时均会调用决策服务。

4. **注册表服务（Registry service）**。独立的 Go 或 TypeScript 服务，定期轮询已注册服务器的 `.well-known/mcp-capabilities` 端点，使用 JSON Schema 进行验证，并提供列表/搜索/验证/启用-禁用功能的用户界面。

5. **能力清单（Capability manifest）**。每个服务器均公开 `.well-known/mcp-capabilities` 端点，包含：工具列表、身份验证要求、传输 URL、负责团队及服务等级目标（SLO）。

6. **破坏性工具隔离（Destructive tool separation）**。会修改状态的工具（如 Jira 创建、Linear 创建、Postgres 写入）部署在第二个 MCP 服务器上，并采用更严格的身份验证流程：令牌必须包含 `approved:by:human` 作用域，且需在 15 分钟内通过 Slack 卡片完成权限提升。

7. **审计日志（Audit log）**。为每个租户维护仅追加的 JSONL 文件：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。在写入前通过 Presidio 进行 PII 脱敏。

8. **负载测试（Load test）**。在 StreamableHTTP 上模拟 100 个并发客户端。通过添加第二个副本演示水平扩展（horizontal scaling）；展示负载均衡器在无会话粘性（session stickiness）的情况下重新分配流量。

9. **一致性测试（Conformance tests）**。针对两台服务器运行官方 MCP 一致性测试套件（MCP conformance suite）。通过所有强制性测试项。

## 使用示例

$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }

## 交付成果

`outputs/skill-mcp-server.md` 描述了交付物内容。这是一个面向内部工具的生产级 MCP 服务器 + 注册表 + 审计层，集成了 OAuth 2.1 作用域与 OPA 网关控制。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 规范一致性 | StreamableHTTP + 能力清单通过 MCP 一致性测试 |
| 20 | 安全性 | 作用域强制校验、覆盖所有工具的 OPA 策略、密钥管理规范 |
| 20 | 可观测性 | 包含 PII 脱敏的逐次工具调用审计日志 |
| 20 | 扩展性 | 100 客户端负载测试的水平扩展演示 |
| 15 | 注册表用户体验 | 发现/验证/启用-禁用工作流 |
| **100** | | |

## 练习

1. 新增一个工具（Confluence 搜索）。通过注册表验证流程（Registry Validation Flow）将其发布，全程无需修改核心服务器（Core Server）。

2. 编写一条开放策略代理（Open Policy Agent, OPA）策略，用于对包含 `email`、`ssn` 或 `phone` 列的 PostgreSQL 查询结果进行数据脱敏。通过探测查询（Probe Query）进行验证测试。

3. 针对本地延迟（Local Latency）对流式 HTTP（StreamableHTTP）与标准输入输出（stdio）进行基准测试（Benchmark）。报告单次调用的 p50/p95 延迟数据。

4. 实现按租户配额（Per-Tenant Quota）机制：限制每个租户下的每个工具每分钟最多调用 N 次。通过第二条 OPA 策略强制执行该限制。

5. 运行 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 仓库中的 MCP 一致性测试套件（MCP Conformance Suite），并修复所有未通过的测试用例。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| StreamableHTTP | “2026 MCP 传输协议” | 无状态 HTTP + 流式传输；用于替代网络服务器的 SSE + stdio |
| Capability manifest | “标准路径文档” | 包含工具列表、认证信息和传输 URL 的 `.well-known/mcp-capabilities` 文件 |
| OPA / Rego | “策略引擎” | 用于根据外部规则授权工具调用的 Open Policy Agent（开放策略代理） |
| Scope elevation | “人工审批” | 通过 Slack 审批授予的短期作用域权限，执行破坏性工具时必需 |
| Registry | “工具发现” | 根据能力清单（Capability Manifest）对 MCP 服务器建立索引的服务 |
| Workload identity | “SPIFFE / SPIRE” | 用于签发 OAuth 令牌的加密服务身份标识 |
| Conformance suite | “规范测试” | 用于验证 StreamableHTTP 与工具清单正确性的官方 MCP 测试套件 |

## 延伸阅读

- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据（Capability Metadata）、注册表（Registry）
- [AAIF MCP 注册表规范](https://github.com/modelcontextprotocol/registry) — 2026 版注册表规范
- [AWS ECS 参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考生产环境部署方案
- [Pinterest 内部 MCP 生态](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署案例
- [Block `goose` MCP 使用](https://block.github.io/goose/) — 参考智能体（Agent）消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架
- [Open Policy Agent](https://www.openpolicyagent.org/) — 策略引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) — 工作负载身份（Workload Identity）参考